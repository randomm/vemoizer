"""CLI tests for the ``meeting`` command's M3 split-recording grouping
(issue #87).

Every test monkeypatches ``HOME`` and chdirs into a tmp dir
(``isolate_home``) and fakes ``transcribe_file`` / ``decode_boundaries``
— no models, no network, no ffmpeg. The expert ``transcribe`` command's
grouping contracts live in ``tests/test_grouping.py`` /
``tests/test_grouping_bugs.py``; this file pins the same contracts on
the ``meeting`` preset path (via ``run_preset`` → ``run_batch``) plus
the dated-name-from-mtime naming.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app

runner = CliRunner()

# A tail with no closing cue and a head that continues the sentence ->
# propose_groups classifies the boundary as a continuation (one group).
_MID_SENTENCE_TAIL = "ja tässä ollaan nyt siinä vaiheessa missä"
_CONT_HEAD = "tässä jatketaan seuraavaksi kohdassa"

# A tail with a closing cue -> propose_groups classifies the boundary
# as a break (two single-file groups).
_CLOSING_TAIL = "kiitoksia kaikille, moi"


def _touch(names: list[str], tmp_path: Path) -> list[Path]:
    files = [tmp_path / n for n in names]
    for f in files:
        f.touch()
    return files


def _set_mtime(path: Path, y: int, m: int, d: int) -> None:
    """Set *path*'s mtime to a fixed local date (YYYY-MM-DD)."""
    ts = time.mktime((y, m, d, 12, 0, 0, 0, 0, 0))
    os.utime(path, (ts, ts))


def _fake_group_seams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tail: str = _MID_SENTENCE_TAIL,
    head: str = _CONT_HEAD,
) -> dict:
    """Monkeypatch decode_boundaries, concat_groups, part_offsets, and
    _decode_edge_window so the grouping flow never touches ffmpeg.

    Returns a ``touched`` dict for assertions.
    """
    import vemoizer.grouping as grouping

    touched = {"decode": 0, "concat": 0, "offsets": 0, "edge": 0}

    def fake_decode_boundaries(files, transcribe_fn=None):
        touched["decode"] += 1
        return [tail], [head]

    def fake_concat(files):
        touched["concat"] += 1
        return files[0]

    def fake_offsets(files):
        touched["offsets"] += 1
        return []

    def fake_edge(path, start, end):
        touched["edge"] += 1
        return None

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "concat_groups", fake_concat)
    monkeypatch.setattr(grouping, "part_offsets", fake_offsets)
    monkeypatch.setattr(grouping, "_decode_edge_window", fake_edge)
    # Also patch the batch-level re-exports (run_batch resolves via batch).
    import vemoizer.batch as batch

    monkeypatch.setattr(batch, "concat_groups", fake_concat)
    monkeypatch.setattr(batch, "part_offsets", fake_offsets)
    return touched


def test_meeting_yes_accepts_proposal_one_group_one_pair(tmp_path, monkeypatch) -> None:
    """meeting a.m4a b.m4a --yes: one continuation proposal is accepted,
    one group -> ONE transcribe call and one dated .md + .json pair in
    the CWD. CliRunner's stdin is non-TTY; --yes passes the TTY guard."""
    import vemoizer.pipeline as pipeline_module

    transcribed: list[str] = []

    def fake_transcribe(path, **kwargs):
        transcribed.append(Path(path).name)
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Team Sync"},
        }

    _fake_group_seams(monkeypatch, tmp_path)
    _touch(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert len(transcribed) == 1
    md_files = list(tmp_path.glob("*.md"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(md_files) == 1, f"expected 1 .md, got {md_files}"
    assert len(json_files) == 1, f"expected 1 .json, got {json_files}"
    assert "Team Sync" in md_files[0].name


def test_meeting_yes_break_yields_one_pair_per_group(tmp_path, monkeypatch) -> None:
    """meeting a.m4a b.m4a --yes where the boundary is a break: two
    single-file groups -> two dated .md/.json pairs, no combined pair."""
    import vemoizer.pipeline as pipeline_module

    transcribed: list[str] = []

    def fake_transcribe(path, **kwargs):
        transcribed.append(Path(path).name)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": Path(path).stem.upper()},
        }

    _fake_group_seams(monkeypatch, tmp_path, tail=_CLOSING_TAIL, head="")
    _touch(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert sorted(transcribed) == ["a.m4a", "b.m4a"]
    assert len(list(tmp_path.glob("*.md"))) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_meeting_no_group_transcribes_each_file_standalone(
    tmp_path, monkeypatch
) -> None:
    """--no-group: each file transcribed standalone (today's behaviour) —
    no boundary decode, one dated pair per file."""
    import vemoizer.pipeline as pipeline_module

    touched = _fake_group_seams(monkeypatch, tmp_path)
    transcribed: list[str] = []

    def fake_transcribe(path, **kwargs):
        transcribed.append(Path(path).name)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": Path(path).stem.upper()},
        }

    _touch(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--no-group"])
    assert result.exit_code == 0
    assert touched["decode"] == 0
    assert sorted(transcribed) == ["a.m4a", "b.m4a"]
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_meeting_non_tty_without_yes_or_no_group_fails_immediately(
    tmp_path, monkeypatch
) -> None:
    """2+ files, non-TTY, neither flag: exit 2 BEFORE any boundary
    decode or transcribe — same contract as ``transcribe``."""
    import vemoizer.ingest as ingest
    import vemoizer.pipeline as pipeline_module

    touched = _fake_group_seams(monkeypatch, tmp_path)
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

    _touch(["a.m4a", "b.m4a"], tmp_path)
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


def test_meeting_yes_and_no_group_mutually_excluded(tmp_path, monkeypatch) -> None:
    """--yes and --no-group together: exit 2, "mutually exclusive",
    before any decode — even for a single file (the check fires before
    the single-file short-circuit, matching run_batch's existing order)."""
    _fake_group_seams(monkeypatch, tmp_path)
    _touch(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes", "--no-group"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
    # Single file too: the flag check fires before the short-circuit.
    result = runner.invoke(app, ["meeting", "a.m4a", "--yes", "--no-group"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr


def test_meeting_single_file_needs_no_tty(tmp_path, monkeypatch) -> None:
    """A single-file meeting run performs NO grouping work: no boundary
    decode, no TTY requirement — it flows straight to the plain loop."""
    import vemoizer.pipeline as pipeline_module

    touched = _fake_group_seams(monkeypatch, tmp_path)
    transcribe_calls = 0
    isatty_calls = 0

    def fake_transcribe(path, **kwargs):
        nonlocal transcribe_calls
        transcribe_calls += 1
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    def _counting_isatty() -> bool:
        nonlocal isatty_calls
        isatty_calls += 1
        return False

    _touch(["a.m4a"], tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr("sys.stdin.isatty", _counting_isatty)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert touched["decode"] == 0
    assert transcribe_calls == 1
    assert isatty_calls == 0


def test_meeting_grouped_run_passes_preset_options(tmp_path, monkeypatch) -> None:
    """The grouped path must pass the meeting preset's RunOptions through
    to transcribe_file: profile=meeting, diarize=True, repair=True,
    speakers (2, 6) by default."""
    import vemoizer.pipeline as pipeline_module

    _fake_group_seams(monkeypatch, tmp_path)
    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    _touch(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert seen["profile"] == "meeting"
    assert seen["diarize"] is True
    assert seen["repair"] is True
    assert seen["speakers"] == (2, 6)


def test_meeting_dated_name_uses_file_mtime(tmp_path, monkeypatch) -> None:
    """The output date prefix is the file's MODIFICATION date, not today."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {"text": "moikka", "segments": [], "notes": {"title": "Team Sync"}}

    files = _touch(["a.m4a"], tmp_path)
    _set_mtime(files[0], 2025, 3, 14)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    assert md_files[0].name.startswith("2025-03-14 Team Sync"), md_files[0].name


def test_meeting_group_dated_name_uses_first_part_mtime(tmp_path, monkeypatch) -> None:
    """A group's date comes from the FIRST part's mtime (natural-sort
    order), not today and not the later part's mtime."""
    import vemoizer.pipeline as pipeline_module

    _fake_group_seams(monkeypatch, tmp_path)

    def fake_transcribe(path, **kwargs):
        return {"text": "moikka", "segments": [], "notes": {"title": "Team Sync"}}

    files = _touch(
        ["Uusi \u00e4\u00e4nitys 1.m4a", "Uusi \u00e4\u00e4nitys 2.m4a"], tmp_path
    )
    _set_mtime(files[0], 2025, 3, 14)
    _set_mtime(files[1], 2025, 6, 1)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(
        app,
        [
            "meeting",
            "Uusi \u00e4\u00e4nitys 1.m4a",
            "Uusi \u00e4\u00e4nitys 2.m4a",
            "--yes",
        ],
    )
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    assert md_files[0].name.startswith("2025-03-14 Team Sync"), md_files[0].name


def test_meeting_unreadable_mtime_falls_back_to_today(tmp_path, monkeypatch) -> None:
    """When the file cannot be stat'ed, the date falls back to today
    (clean fallback, never an exception)."""
    from datetime import date

    import vemoizer.batch_preset as batch_preset
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    # Verify _mtime_date_str falls back to today on OSError.
    def boom_stat(path, *a, **kw):
        raise OSError("no such file")

    real_stat = os.stat
    monkeypatch.setattr(os, "stat", boom_stat)
    try:
        fallback = batch_preset._mtime_date_str(Path("/nonexistent/file"))
    finally:
        monkeypatch.setattr(os, "stat", real_stat)
    assert fallback == date.today().isoformat()

    # Now verify the full run uses today's date (default, no mtime patch
    # during the run — the fallback was verified above).
    _touch(["a.m4a"], tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    # The date is the file's mtime (just set by touch = today), so the
    # prefix is today's date.
    assert md_files[0].name.startswith(date.today().isoformat())
