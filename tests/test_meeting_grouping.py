"""CLI tests for the ``meeting`` command's M3 split-recording grouping
(issue #87).

Every test monkeypatches ``HOME`` and chdirs into a tmp dir
(``isolate_home``) and fakes ``transcribe_file`` / ``decode_boundaries``
— no models, no network, no ffmpeg. The expert ``transcribe`` command's
grouping contracts live in ``tests/test_grouping.py`` /
``tests/test_grouping_bugs.py``; this file pins the same contracts on
the ``meeting`` preset path (via ``run_preset`` → ``run_batch``), the
dated-name-from-mtime naming, and the temp-glossary lifecycle.
"""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pytest
from _cli_helpers import fake_transcribe, isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.grouping as grouping
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()

# A tail with no closing cue and a head that continues the sentence ->
# propose_groups classifies the boundary as a continuation (one group).
_MID_SENTENCE_TAIL = "ja tässä ollaan nyt siinä vaiheessa missä"
_CONT_HEAD = "tässä jatketaan seuraavaksi kohdassa"

# A tail with a closing cue -> propose_groups classifies the boundary
# as a break (two single-file groups).
_CLOSING_TAIL = "kiitoksia kaikille, moi"

# Distinct mtimes for the parts: the group's date must come from the
# FIRST part in natural-sort order (2025-03-14), not from the later
# part (2025-06-01).
_PART_A = "Uusi \u00e4\u00e4nitys 1.m4a"
_PART_B = "Uusi \u00e4\u00e4nitys 2.m4a"
_DATE_A = "2025-03-14"
_DATE_B = "2025-06-01"


def _set_mtime(path: Path, y: int, m: int, d: int) -> None:
    """Set *path*'s mtime to a fixed local date (YYYY-MM-DD)."""
    ts = time.mktime((y, m, d, 12, 0, 0, 0, 0, 0))
    os.utime(path, (ts, ts))


def _fake_group_seams(
    monkeypatch: pytest.MonkeyPatch,
    tail: str = _MID_SENTENCE_TAIL,
    head: str = _CONT_HEAD,
) -> dict:
    """Monkeypatch decode_boundaries, concat_groups, and part_offsets so
    the grouping flow never touches ffmpeg.

    Returns a ``touched`` dict for assertions.
    """
    touched = {"decode": 0, "concat": 0, "offsets": 0}

    def fake_decode_boundaries(files, transcribe_fn=None, **kw):
        touched["decode"] += 1
        return [tail], [head]

    def fake_concat(files, **kw):
        touched["concat"] += 1
        return files[0]

    def fake_offsets(files, **kw):
        touched["offsets"] += 1
        return []

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "concat_groups", fake_concat)
    monkeypatch.setattr(grouping, "part_offsets", fake_offsets)
    # Also patch the batch-level re-exports (run_batch resolves via batch).
    import vemoizer.batch as batch

    monkeypatch.setattr(batch, "concat_groups", fake_concat)
    monkeypatch.setattr(batch, "part_offsets", fake_offsets)
    return touched


def _pair_files(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    return sorted(tmp_path.glob("*.md")), sorted(tmp_path.glob("*.json"))


# --- (a) meeting --help lists the grouping flags -------------------------


def test_meeting_help_lists_yes_and_no_group() -> None:
    """``meeting --help`` must list --yes and --no-group (issue #87)."""
    result = runner.invoke(app, ["meeting", "--help"])
    assert result.exit_code == 0
    assert "--yes" in result.stdout
    assert "--no-group" in result.stdout


# --- (b) 2 files + --yes: one continuation, ONE pair, no TTY --------------


def test_meeting_yes_accepts_proposal_one_group_one_pair(tmp_path, monkeypatch) -> None:
    """meeting a.m4a b.m4a --yes: one continuation proposal is accepted,
    one group -> ONE transcribe call and one dated .md + .json pair in
    the CWD. CliRunner's stdin is non-TTY; --yes passes the TTY guard."""
    _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert len(record) == 1
    mds, js = _pair_files(tmp_path)
    assert len(mds) == 1, f"expected 1 .md, got {mds}"
    assert len(js) == 1, f"expected 1 .json, got {js}"
    assert "Team Sync" in mds[0].name


# --- (c) non-TTY, no flags: exit 2 before any decode ----------------------


def test_meeting_non_tty_without_yes_or_no_group_fails_immediately(
    tmp_path, monkeypatch
) -> None:
    """2+ files, non-TTY, neither flag: exit 2 BEFORE any boundary
    decode or transcribe — same contract as ``transcribe``. The guard
    test uses the default input_fn=None and tty_isatty False."""
    import vemoizer.ingest as ingest

    touched = _fake_group_seams(monkeypatch)
    transcribe_calls = 0
    ingest_calls = 0

    def fake_transcribe(path, **kwargs):
        nonlocal transcribe_calls
        transcribe_calls += 1
        return {"text": "hei", "segments": []}

    def fake_ingest_audio(path):
        nonlocal ingest_calls
        ingest_calls += 1
        return None

    touch_files(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(ingest, "ingest_audio", fake_ingest_audio)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a"])
    assert result.exit_code == 2
    assert "--yes" in result.stderr
    assert "--no-group" in result.stderr
    assert "TTY" in result.stderr
    assert touched["decode"] == 0
    assert transcribe_calls == 0
    assert ingest_calls == 0


# --- (d) --yes + --no-group: mutually exclusive, also single file ----------


def test_meeting_yes_and_no_group_mutually_excluded(tmp_path, monkeypatch) -> None:
    """--yes and --no-group together: exit 2, "mutually exclusive",
    before any decode — even for a single file (the check fires before
    the single-file short-circuit, matching run_batch's existing order)."""
    touched = _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes", "--no-group"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
    # Single file too: the flag check fires before the short-circuit.
    result = runner.invoke(app, ["meeting", "a.m4a", "--yes", "--no-group"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
    assert touched["decode"] == 0


# --- (e) --no-group: two independent transcribes, no boundary decode -------


def test_meeting_no_group_transcribes_each_file_standalone(
    tmp_path, monkeypatch
) -> None:
    """--no-group: each file transcribed standalone (today's behaviour) —
    no boundary decode, one dated pair per file with today's date."""
    touched = _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--no-group"])
    assert result.exit_code == 0
    assert touched["decode"] == 0
    assert sorted(record) == ["a.m4a", "b.m4a"]
    mds, _ = _pair_files(tmp_path)
    assert len(mds) == 2
    today = date.today().isoformat()
    assert all(m.name.startswith(today) for m in mds), mds


# --- (f) break proposal -> two groups -> two dated pairs --------------------


def test_meeting_yes_break_yields_one_pair_per_group(tmp_path, monkeypatch) -> None:
    """meeting a.m4a b.m4a --yes where the boundary is a break: two
    single-file groups -> two dated .md/.json pairs, no combined pair."""
    _fake_group_seams(monkeypatch, tail=_CLOSING_TAIL, head="")
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert sorted(record) == ["a.m4a", "b.m4a"]
    mds, js = _pair_files(tmp_path)
    assert len(mds) == 2
    assert len(js) == 2


# --- (g) single file, no flags, non-TTY: no grouping, no prompt -------------


def test_meeting_single_file_needs_no_tty(tmp_path, monkeypatch) -> None:
    """A single-file meeting run performs NO grouping work: no boundary
    decode, no TTY requirement, no prompt — it flows straight to the
    plain loop and yields one dated pair."""
    touched = _fake_group_seams(monkeypatch)
    isatty_calls = 0

    def _counting_isatty() -> bool:
        nonlocal isatty_calls
        isatty_calls += 1
        return False

    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="T")
    monkeypatch.setattr("sys.stdin.isatty", _counting_isatty)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert touched["decode"] == 0
    assert len(record) == 1
    assert isatty_calls == 0
    mds, _ = _pair_files(tmp_path)
    assert len(mds) == 1


# --- (h) meeting-preset RunOptions reach transcribe_file --------------------


def test_meeting_grouped_run_passes_preset_options(tmp_path, monkeypatch) -> None:
    """The grouped path must pass the meeting preset's RunOptions through
    to transcribe_file: profile=meeting, diarize=True, repair=True,
    speakers (2, 6) by default."""
    _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert seen["profile"] == "meeting"
    assert seen["diarize"] is True
    assert seen["repair"] is True
    assert seen["speakers"] == (2, 6)


def test_meeting_grouped_no_diarize_flag_reaches_transcribe(
    tmp_path, monkeypatch
) -> None:
    """--no-diarize overrides the meeting preset default: the grouped
    path must pass diarize=False through to transcribe_file."""
    _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes", "--no-diarize"])
    assert result.exit_code == 0
    assert seen["diarize"] is False


# --- (i) dated name from mtime ----------------------------------------------


def test_meeting_dated_name_uses_file_mtime(tmp_path, monkeypatch) -> None:
    """The output date prefix is the file's MODIFICATION date, not today."""

    def fake_transcribe(path, **kwargs):
        return {"text": "moikka", "segments": [], "notes": {"title": "Team Sync"}}

    files = touch_files(["a.m4a"], tmp_path)
    _set_mtime(files[0], 2025, 3, 14)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    mds, _ = _pair_files(tmp_path)
    assert len(mds) == 1
    assert mds[0].name.startswith("2025-03-14 Team Sync"), mds[0].name


def test_meeting_group_dated_name_uses_first_part_mtime_non_sorted_args(
    tmp_path, monkeypatch
) -> None:
    """A group's date comes from the FIRST part's mtime (natural-sort
    order). The parts get distinct mtimes and are passed in NON-sorted
    argument order (B before A); the date must still be the
    natural-sort-first part's (A's 2025-03-14), not B's 2025-06-01."""
    _fake_group_seams(monkeypatch)

    def fake_transcribe(path, **kwargs):
        return {"text": "moikka", "segments": [], "notes": {"title": "Team Sync"}}

    files = touch_files([_PART_A, _PART_B], tmp_path)
    _set_mtime(files[0], 2025, 3, 14)
    _set_mtime(files[1], 2025, 6, 1)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    # Non-sorted argument order: B first, then A.
    result = runner.invoke(app, ["meeting", _PART_B, _PART_A, "--yes"])
    assert result.exit_code == 0
    mds, _ = _pair_files(tmp_path)
    assert len(mds) == 1
    assert mds[0].name.startswith("2025-03-14 Team Sync"), mds[0].name


def test_meeting_mtime_date_str_falls_back_to_today(monkeypatch) -> None:
    """When the file cannot be stat'ed, _mtime_date_str falls back to
    today (clean fallback, never an exception)."""
    import vemoizer.batch_preset as batch_preset

    def boom_stat(path, *a, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(batch_preset.os, "stat", boom_stat)
    assert (
        batch_preset._mtime_date_str(Path("/nonexistent/file"))
        == date.today().isoformat()
    )


def test_meeting_unreadable_mtime_falls_back_to_today(tmp_path, monkeypatch) -> None:
    """Full-run version: the source file is deleted before the write,
    so the dated pair gets today's date (the fallback), not a crash.
    The unit-level fallback is pinned by
    test_meeting_mtime_date_str_falls_back_to_today."""
    import vemoizer.batch_preset as batch_preset

    touch_files(["a.m4a"], tmp_path)
    isolate_home(monkeypatch, tmp_path, tmp_path)

    state = {"deleted": False}
    real_mtime = batch_preset._mtime_date_str

    def mtime_after_transcribe(p: Path) -> str:
        # The source was removed mid-run: the write must hit the
        # today-fallback branch (real os.stat on the gone file).
        assert state["deleted"], "file deleted before the write seam ran"
        return real_mtime(p)

    def fake_transcribe(path, **kw):
        state["deleted"] = True
        Path(path).unlink(missing_ok=True)
        return {"text": "moikka", "segments": [], "notes": {"title": "Team Sync"}}

    monkeypatch.setattr(batch_preset, "_mtime_date_str", mtime_after_transcribe)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    mds, _ = _pair_files(tmp_path)
    assert len(mds) == 1
    assert mds[0].name.startswith(date.today().isoformat()), mds[0].name


# --- (j) temp glossary deleted on success / failure / KeyboardInterrupt -----


def _write_glossary_layer(tmp_path: Path, line: str = "Nordea\n") -> None:
    """Write a project glossary layer so the temp file IS created
    (the cleanup paths are exercised, not the no-layer trivial case)."""
    (tmp_path / ".vemoizer").mkdir(exist_ok=True)
    (tmp_path / ".vemoizer" / "glossary.txt").write_text(line, encoding="utf-8")


def _leftover_glossary(tmp_path: Path) -> list[Path]:
    return [p for p in tmp_path.iterdir() if p.name.startswith("vemoizer-")]


def test_meeting_grouped_success_deletes_temp_glossary(tmp_path, monkeypatch) -> None:
    """Grouped success: the composed-glossary temp file is deleted after
    the run (the finally)."""
    _write_glossary_layer(tmp_path)
    _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert _leftover_glossary(tmp_path) == []


def test_meeting_grouped_failure_deletes_temp_glossary(tmp_path, monkeypatch) -> None:
    """A failing group (the fake transcribe raises RuntimeError): exit
    1, one clean error line, and the temp glossary still deleted."""
    import vemoizer.batch as batch

    _write_glossary_layer(tmp_path)
    _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)

    def fake_transcribe(path, **kwargs):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(batch, "_resolve_llm_config", lambda p: None)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert _leftover_glossary(tmp_path) == []


def test_meeting_grouped_keyboard_interrupt_deletes_temp_glossary(
    tmp_path, monkeypatch
) -> None:
    """A KeyboardInterrupt from the fake transcribe must propagate AND
    the temp glossary must still be deleted (the finally)."""
    import vemoizer.batch as batch

    _write_glossary_layer(tmp_path)
    _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)

    def fake_transcribe(path, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(batch, "_resolve_llm_config", lambda p: None)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    # Typer converts the KeyboardInterrupt into exit 130 (SIGINT code);
    # CliRunner catches it rather than letting it propagate.
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 130
    assert _leftover_glossary(tmp_path) == []


def test_meeting_no_group_deletes_temp_glossary(tmp_path, monkeypatch) -> None:
    """--no-group plain loop: the temp glossary is deleted on success
    (the finally) — unchanged contract from issue #82."""
    _write_glossary_layer(tmp_path)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--no-group"])
    assert result.exit_code == 0
    assert _leftover_glossary(tmp_path) == []


# --- (k) memo unchanged ------------------------------------------------------


def test_memo_two_files_stay_per_file_no_grouping(tmp_path, monkeypatch) -> None:
    """memo with 2 files stays per-file: two independent transcribes,
    two dated pairs, no boundary decode, no grouping flags. The
    grouped-path seams must all stay untouched."""
    touched = _fake_group_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "b.m4a"])
    assert result.exit_code == 0
    assert touched["decode"] == 0
    assert sorted(record) == ["a.m4a", "b.m4a"]
    mds, _ = _pair_files(tmp_path)
    assert len(mds) == 2


def test_memo_help_has_no_grouping_flags() -> None:
    """memo stays per-file: no --yes / --no-group in memo --help
    (grouping is a meeting-only feature, issue #87)."""
    result = runner.invoke(app, ["memo", "--help"])
    assert result.exit_code == 0
    assert "--yes" not in result.stdout
    assert "--no-group" not in result.stdout
