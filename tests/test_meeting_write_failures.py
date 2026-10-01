"""Write-seam failure contracts for the meeting preset (PR #88, issue #87).

Two real defects fixed here:

1. **Preset write step reporting failure**: ``_write_preset_output`` can
   write the ``.md`` and then fail on the ``.json`` (ENOSPC, EROFS, a
   deleted CWD) — ``_write_output`` prints ``error: could not write ...``
   and returns False, so the returned list has one fewer path. The run
   must exit 1 for that file/group in EVERY loop (single file,
   ``--no-group``, grouped), never claim a pair that was not fully
   written in the ``wrote`` summary, and keep writing later files/groups.
2. **The ``write_group_fn`` seam in ``run_batch`` was unguarded**: an
   unexpected exception there escaped as a raw traceback after minutes of
   decoding. ``run_batch``'s contract is a clean one-line error per
   group, keep going, exit 1 if any group failed.

Pure stdlib: fakes patch ``pipeline.transcribe_file`` and the grouping
seams — no models, no network, no ffmpeg. Every test isolates HOME and
chdirs into its own tmp dir (``isolate_home`` or ``monkeypatch.chdir``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import fake_transcribe, isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.batch as batch
import vemoizer.batch_preset as batch_preset
import vemoizer.grouping as grouping
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.presets import RunOptions

runner = CliRunner()


def _fake_continuation_seams(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the grouping seams so 2 files become ONE multi-part group."""
    touched = {"decode": 0, "concat": 0, "offsets": 0}

    def fake_decode_boundaries(files, transcribe_fn=None):
        touched["decode"] += 1
        return ["ja tässä ollaan nyt siinä vaiheessa missä"], ["tässä jatketaan"]

    def fake_concat(files):
        touched["concat"] += 1
        return files[0]

    def fake_offsets(files):
        touched["offsets"] += 1
        return []

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "concat_groups", fake_concat)
    monkeypatch.setattr(grouping, "part_offsets", fake_offsets)
    monkeypatch.setattr(batch, "concat_groups", fake_concat)
    monkeypatch.setattr(batch, "part_offsets", fake_offsets)
    return touched


def _fake_break_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the grouping seams so 2 files become TWO single-file groups
    (a closing cue on the boundary -> a break)."""

    def fake_decode_boundaries(files, transcribe_fn=None):
        return ["kiitoksia kaikille, moi"], ["uusi aloitus tässä"]

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    monkeypatch.setattr(batch, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(batch, "part_offsets", lambda files: [])


def _real_partial_write(result, first_stem, out_dir, *, date_str=None):
    """The real pair write, but the ``.json`` (the last format) is forced
    to fail: its target's parent is a FILE, so ``_write_output`` raises
    ``OSError`` (IsADirectoryError), prints the clean
    ``error: could not write ...`` line, and skips the json. The ``.md``
    is written to the real path."""
    from vemoizer.batch_output import PRESET_FORMATS, _write_output
    from vemoizer.output.naming import collision_free_paths

    base = batch_output_dated_base(result, first_stem, date_str)
    paths = collision_free_paths(out_dir, base, [f".{fmt}" for fmt in PRESET_FORMATS])
    written: list[str] = []
    (out_dir / "block_file").write_text("x", encoding="utf-8")
    for path, fmt in zip(paths, PRESET_FORMATS, strict=True):
        if fmt == "json":
            _write_output(out_dir / "block_file" / path.name, result, fmt)
            continue
        if _write_output(path, result, fmt):
            written.append(path.name)
    (out_dir / "block_file").unlink(missing_ok=True)
    return written


def batch_output_dated_base(result, first_stem, date_str) -> str:
    from vemoizer.batch_output import dated_basename

    return dated_basename(
        str(result.get("notes", {}).get("title", "")),
        fallback_stem=first_stem,
        date_str=date_str,
    )


def _wrote_names(stdout: str) -> list[str]:
    """The file names from the ``wrote <path>`` summary lines in *stdout*."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith("wrote ")]
    return [ln[len("wrote ") :] for ln in lines]


def _options() -> RunOptions:
    return RunOptions.expert_transcribe(
        profile="meeting",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )


# --- (a) _write_preset_output partial failure: exit 1, pair not claimed ---


def test_grouped_write_failure_exits_1_and_keeps_going(
    tmp_path, monkeypatch, capsys
) -> None:
    """Grouped run: the group's .json write fails (the real
    ``_write_output`` error line is printed, the .md exists), the run
    exits 1, and the 'wrote' summary claims only files that really
    exist. Two files -> one group (continuation)."""
    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    monkeypatch.setattr(batch_preset, "_write_preset_output", _real_partial_write)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 1
    assert "error: could not write" in result.stderr
    mds = sorted(tmp_path.glob("*.md"))
    js = sorted(tmp_path.glob("*.json"))
    assert len(mds) == 1, mds
    assert len(js) == 0, js
    wrote_lines = _wrote_names(result.stdout)
    for name in wrote_lines:
        assert (tmp_path / name).exists(), f"claimed but missing: {name}"
    assert any(name.endswith(".md") for name in wrote_lines)
    assert not any(name.endswith(".json") for name in wrote_lines)


def test_plain_loop_write_failure_exits_1(tmp_path, monkeypatch) -> None:
    """--no-group plain loop: file 1's pair is half-written (md yes, json
    no), file 2 still writes its full pair; exit 1; the summary never
    claims the missing json."""
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    calls = {"n": 0}

    def partial(result, first_stem, out_dir, *, date_str=None):
        # Only the first call (file 1) fails the json write; file 2 is fine.
        calls["n"] += 1
        if calls["n"] == 1:
            return _real_partial_write(result, first_stem, out_dir, date_str=date_str)
        return batch._write_preset_output(
            result, first_stem, out_dir, date_str=date_str
        )

    monkeypatch.setattr(batch_preset, "_write_preset_output", partial)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--no-group"])
    assert result.exit_code == 1
    assert "error: could not write" in result.stderr
    mds = sorted(tmp_path.glob("*.md"))
    js = sorted(tmp_path.glob("*.json"))
    assert len(mds) == 2, mds
    assert len(js) == 1, js
    # The summary claims exactly the 3 files that exist (2 md + 1 json)
    # and never the missing json.
    wrote_lines = _wrote_names(result.stdout)
    assert len(wrote_lines) == 3
    for name in wrote_lines:
        assert (tmp_path / name).exists(), f"claimed but missing: {name}"
    assert sum(1 for n in wrote_lines if n.endswith(".json")) == 1


def test_single_file_write_failure_exits_1(tmp_path, monkeypatch) -> None:
    """Single file: the .md is written, the .json fails; exit 1, one
    error line, the md claimed in the summary, the json not."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    monkeypatch.setattr(batch_preset, "_write_preset_output", _real_partial_write)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert "error: could not write" in result.stderr
    mds = sorted(tmp_path.glob("*.md"))
    js = sorted(tmp_path.glob("*.json"))
    assert len(mds) == 1, mds
    assert len(js) == 0, js
    wrote_lines = _wrote_names(result.stdout)
    assert len(wrote_lines) == 1
    assert wrote_lines[0].endswith(".md")
    assert (tmp_path / wrote_lines[0]).exists()


def test_normal_run_still_exits_0_with_both_files(tmp_path, monkeypatch) -> None:
    """A normal run (no write failure) still exits 0 with both the .md
    and the .json written and claimed."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    mds = sorted(tmp_path.glob("*.md"))
    js = sorted(tmp_path.glob("*.json"))
    assert len(mds) == 1
    assert len(js) == 1
    wrote_lines = _wrote_names(result.stdout)
    assert len(wrote_lines) == 2


# --- (b) the write_group_fn seam in run_batch: unguarded exception ---------


def test_seam_runtime_error_is_clean_and_continues(
    tmp_path, monkeypatch, capsys
) -> None:
    """A seam that raises RuntimeError for the FIRST of two groups:
    exit 1, one clean error line (no traceback), and the SECOND group
    still written (its real pair is on disk)."""
    _fake_break_seams(monkeypatch)
    files = touch_files(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.chdir(tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)

    calls = {"n": 0}

    def raising_seam(label, result):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk on fire")
        first = (
            Path(label) if isinstance(label, Path) else Path(str(label).split("+")[0])
        )
        stem, _ = nfc_stem_and_suffix(first)
        return batch._write_preset_output(
            result, stem, Path.cwd(), date_str="2025-01-01"
        )

    code = batch.run_batch(
        files,
        _options(),
        yes=True,
        write_group_fn=raising_seam,
        input_fn=lambda _: "\n",
    )
    out, err = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in err
    assert "could not produce output" in err
    assert "disk on fire" in err
    # Exactly one clean error line for the seam failure.
    assert err.count("could not produce output") == 1
    # The second group was still written (one full pair).
    mds = sorted(tmp_path.glob("*.md"))
    js = sorted(tmp_path.glob("*.json"))
    assert len(mds) == 1, mds
    assert len(js) == 1, js
    # Both groups were attempted (the seam ran twice).
    assert calls["n"] == 2


def test_seam_keyboard_interrupt_still_propagates(tmp_path, monkeypatch) -> None:
    """A KeyboardInterrupt from the seam must propagate (not be swallowed
    as a group failure)."""
    _fake_break_seams(monkeypatch)
    files = touch_files(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_transcribe(path, **kwargs):
        return {"text": "hei", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)

    def write_group(label, result):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(
            files,
            _options(),
            yes=True,
            write_group_fn=write_group,
            input_fn=lambda _: "\n",
        )


def test_seam_system_exit_still_propagates(tmp_path, monkeypatch) -> None:
    """A SystemExit from the seam must propagate."""
    _fake_break_seams(monkeypatch)
    files = touch_files(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_transcribe(path, **kwargs):
        return {"text": "hei", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)

    def write_group(label, result):
        raise SystemExit(7)

    with pytest.raises(SystemExit):
        batch.run_batch(
            files,
            _options(),
            yes=True,
            write_group_fn=write_group,
            input_fn=lambda _: "\n",
        )


def test_seam_date_lookup_failure_message_is_accurate(
    tmp_path, monkeypatch, capsys
) -> None:
    """A seam failure that runs BEFORE the actual file write (the
    mtime-date lookup, not a write) must not be misattributed: the
    error line says ``could not produce output`` with the reason,
    never ``could not write output``; exit 1, no traceback, no files."""
    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)

    def boom(_path):
        raise OverflowError("year out of range")

    monkeypatch.setattr(batch_preset, "_mtime_date_str", boom)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 1
    assert result.stderr.count("could not produce output") == 1
    assert "year out of range" in result.stderr
    assert "could not write output" not in result.stderr
    assert "Traceback" not in result.stderr
    assert list(tmp_path.glob("*.md")) == []
    assert list(tmp_path.glob("*.json")) == []


def test_seam_none_return_is_backward_compatible(tmp_path, monkeypatch) -> None:
    """The seam may return a bool/None; run_batch must not care (the
    seam's write side effects are what matter) — exit 0 on success."""
    _fake_break_seams(monkeypatch)
    files = touch_files(["a.m4a"], tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_transcribe(path, **kwargs):
        return {"text": "hei", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)

    seen: list[str] = []

    def write_group(label, result):
        seen.append(Path(label).name)
        return None

    code = batch.run_batch(files, _options(), write_group_fn=write_group)
    assert code == 0
    assert seen == ["a.m4a"]
