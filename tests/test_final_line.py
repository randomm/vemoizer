"""Tests for the final "complete" line of a meeting/memo run (issue #148).

The word "complete" appears exactly once per run, on the final line,
after the "wrote ..." lines. Per-stage markers use the stage name
("decode ✓", "diarize ✓", ...), so "complete" on the final line is
unambiguous: the run is done and its files are on disk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
from _cli_helpers import isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.batch as batch_module
import vemoizer.batch_preset as batch_preset_module
import vemoizer.grouping_decode as grouping_decode_module
import vemoizer.notify as notify_module
import vemoizer.pipeline as pipeline_module
import vemoizer.preflight as preflight_module
from vemoizer.batch_output import PRESET_FORMATS
from vemoizer.cli import app
from vemoizer.preset_final import run_went_full
from vemoizer.preset_interrupt import begin_interrupt_tracking

runner = CliRunner()


def _good_result(paragraphs):
    return {
        "text": " ".join(p["text"] for p in paragraphs),
        "segments": paragraphs,
        "paragraphs": paragraphs,
    }


def _two_label_paragraphs():
    return [
        {"start": 0.0, "end": 2.0, "text": "hei", "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "text": "moro", "speaker": "SPEAKER_01"},
    ]


def _fake_transcribe_rich(monkeypatch: pytest.MonkeyPatch, paragraphs) -> None:
    import vemoizer.pipeline as pipeline_module

    def fake(path, **kwargs):
        return {
            "text": " ".join(p["text"] for p in paragraphs),
            "segments": paragraphs,
            "paragraphs": paragraphs,
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake)


def _stub_run_names(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def fake(path, input_fn=None, tty_isatty=None):
        calls.append(path)

    import vemoizer.names_cli as names_cli

    monkeypatch.setattr(names_cli, "run_names", fake)
    return calls


class TestFinalLine:
    def test_success_prints_single_complete_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """On success, the word 'complete' appears exactly once, on the
        final line, after the wrote lines (issue #148 gate resolution)."""
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(app, ["meeting", "a.m4a", "--yes"], input="y\n")
        assert result.exit_code == 0
        # The word "complete" appears exactly once, on the final line.
        assert result.stdout.count("complete") == 1
        # It appears after the wrote lines.
        wrote_idx = result.stdout.index("wrote ")
        complete_idx = result.stdout.index("complete")
        assert complete_idx > wrote_idx

    def test_final_line_carries_no_literal_markup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The final line prints PLAIN text — rich markup like ``[green]``
        would print verbatim through ``typer.echo`` (issue #148 FIX 2)."""
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(app, ["meeting", "a.m4a", "--yes"], input="y\n")
        assert result.exit_code == 0
        final_line = next(s for s in result.stdout.splitlines() if "complete" in s)
        assert "[green]" not in result.stdout
        assert "[" not in final_line, final_line
        assert "complete" in final_line
        assert "wrote 2 file(s)" in final_line

    def test_quiet_suppresses_final_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """--quiet suppresses the final 'complete' line."""
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(
            app, ["meeting", "a.m4a", "--yes", "--quiet"], input="y\n"
        )
        assert result.exit_code == 0
        assert "complete" not in result.stdout


# -- FIX 1: the final line is gated on the run succeeding for ALL files ---
#
# "complete" appears only when every expected pair was fully written:
# exit code 0 AND len(written) == len(files) * len(PRESET_FORMATS), in
# BOTH run_preset paths (plain per-file loop and _run_preset_groups).
# On any failure (partial pair, quality-check failure) the successful
# files' "wrote" lines still print, but no "complete" line ever does.
# Break-and-fail: dropping the gate (reverting the final-line condition
# to `if not quiet and written`) fails the partial-pair, check-failure,
# and mixed-batch assertions below.


def _plain_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the seams the plain per-file path hits (preflight gate, notify,
    and the config pre-check) so the test reaches the transcribe + write
    seams it exercises. The transcribe seam is the name as LOOKED UP in
    the calling modules' own namespaces (``preset_file_transcribe`` for
    the plain loop, ``batch_guard`` for the group path)."""
    monkeypatch.setattr(preflight_module, "preflight_gate", lambda **kw: None)
    monkeypatch.setattr(batch_module, "_resolve_llm_config", lambda p: None)
    monkeypatch.setattr(notify_module, "notify_write", lambda *a, **kw: None)
    monkeypatch.setattr(notify_module, "notify_result", lambda *a, **kw: None)
    # The transcribe seam is a function-local `from vemoizer.pipeline import
    # transcribe_file` — patch the name in the SOURCE module (where the
    # lazy import looks it up), which every call site resolves through.
    monkeypatch.setattr(pipeline_module, "transcribe_file", _plain_fake_transcribe)


def _plain_fake_transcribe(path, **kwargs):
    return _good_result(_two_label_paragraphs())


def _run_preset_plain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: list[Path],
    *,
    result_per_file: dict[str, Any] | Any,
    quiet: bool = False,
    command: str = "meeting",
) -> int:
    """Run run_preset's plain per-file path with the transcribe seam stubbed;
    returns the exit code."""

    def fake_transcribe(file, options=None, glossary_path=None, **kwargs):
        if callable(result_per_file):
            fn = cast(Callable[[Any], Any], result_per_file)
            return fn(file)
        return result_per_file

    monkeypatch.setattr(batch_preset_module, "_transcribe_preset_file", fake_transcribe)
    _plain_seams(monkeypatch)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    tracker = begin_interrupt_tracking()
    return batch_preset_module.run_preset(
        list(files),
        command=command,
        config_path=None,
        glossary_path=None,
        quiet=quiet,
        yes=True,
        tracker=tracker,
        display=None,
    )


def _group_write_seam(result, first_stem, out_dir, *, date_str=None):
    """Write seam for the group path: md ok, json fails (partial pair)."""
    return [f"2025-01-01 {first_stem}.md"]  # partial: json write "failed"


def _group_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the transcribe seam the group path uses (function-local import
    from ``vemoizer.pipeline``)."""
    monkeypatch.setattr(pipeline_module, "transcribe_file", _plain_fake_transcribe)


def _patch_group_seams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stub the seams the meeting 2+ files path hits before the write seam:
    the preflight gate (the models are not in the cache under the fake home),
    the boundary decodes (the zero-byte fixtures are not real audio), and
    the transcribe fn itself (no real audio to decode — the write seam is
    what the test exercises)."""
    _group_seams(monkeypatch)
    monkeypatch.setattr(preflight_module, "preflight_gate", lambda **kw: None)
    monkeypatch.setattr(
        grouping_decode_module,
        "decode_boundaries",
        lambda *a, **kw: (["", ""], ["", ""]),
    )
    isolate_home(monkeypatch, tmp_path, tmp_path)


class TestFinalLineGatedOnSuccess:
    def test_partial_pair_no_complete_line(self, tmp_path, monkeypatch) -> None:
        """Plain path: the .json write fails (partial pair) -> exit 1, the
        .md "wrote" line still prints, but NO "complete" line (issue #148
        core promise: nothing reads 'complete' until the files are on
        disk)."""
        files = touch_files(["a.m4a"], tmp_path)
        record: dict[str, Any] = {}

        def fake_echo(msg, err: bool = False, **kw):
            if not err:
                record.setdefault("echoes", []).append(str(msg))

        def fake_write(result, stem, out_dir, *, date_str=None):
            return ["2025-01-01 a.md"]  # partial pair: only the .md "wrote"

        _plain_seams(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        tracker = begin_interrupt_tracking()
        with (
            mock.patch.object(batch_preset_module, "_write_preset_output", fake_write),
            mock.patch.object(batch_preset_module.typer, "echo", fake_echo),
        ):
            code = batch_preset_module.run_preset(
                list(files),
                command="meeting",
                config_path=None,
                glossary_path=None,
                quiet=False,
                yes=True,
                tracker=tracker,
                display=None,
            )
        assert code == 1
        echoes = record.get("echoes", [])
        assert any(e.startswith("wrote ") for e in echoes), echoes
        # NO "complete" line on a failed run.
        assert all("complete" not in e for e in echoes), echoes

    def test_check_failure_no_complete_line(self, tmp_path, monkeypatch) -> None:
        """Plain path: a file that fails the quality checks -> exit 1, no
        "complete" line, no "wrote" lines (the file never got written)."""
        files = touch_files(["a.m4a"], tmp_path)
        record: dict[str, Any] = {}

        def fake_echo(msg, err: bool = False, **kw):
            if not err:
                record.setdefault("echoes", []).append(str(msg))

        def bad_transcribe(file, options=None, glossary_path=None, **kwargs):
            return {
                "text": "",  # no transcript -> _check_result fails
                "segments": [],
            }

        _plain_seams(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        tracker = begin_interrupt_tracking()
        with (
            mock.patch.object(
                batch_preset_module, "_transcribe_preset_file", bad_transcribe
            ),
            mock.patch.object(batch_preset_module.typer, "echo", fake_echo),
        ):
            code = batch_preset_module.run_preset(
                list(files),
                command="meeting",
                config_path=None,
                glossary_path=None,
                quiet=False,
                yes=True,
                tracker=tracker,
                display=None,
            )
        assert code == 1
        echoes = record.get("echoes", [])
        assert all("complete" not in e for e in echoes), echoes
        assert all(not e.startswith("wrote ") for e in echoes), echoes

    def test_mixed_batch_one_failed_one_succeeded(self, tmp_path, monkeypatch):
        """Plain path, 2 files: the first succeeds (full pair written), the
        second has a partial pair (its .json write fails) -> exit 1, the
        first file's "wrote" lines still print, but NO "complete" line
        for the mixed batch."""
        files = touch_files(["a.m4a", "b.m4a"], tmp_path)
        record: dict[str, Any] = {}

        def fake_echo(msg, err: bool = False, **kw):
            if not err:
                record.setdefault("echoes", []).append(str(msg))

        def fake_write(result, stem, out_dir, *, date_str=None):
            if stem == "a":
                return ["2025-01-01 a.md", "2025-01-01 a.json"]  # full pair
            return ["2025-01-01 b.md"]  # b: partial pair (json write failed)

        _plain_seams(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        tracker = begin_interrupt_tracking()
        with (
            mock.patch.object(batch_preset_module, "_write_preset_output", fake_write),
            mock.patch.object(batch_preset_module.typer, "echo", fake_echo),
        ):
            code = batch_preset_module.run_preset(
                list(files),
                command="meeting",
                config_path=None,
                glossary_path=None,
                quiet=False,
                yes=True,
                tracker=tracker,
                display=None,
            )
        assert code == 1
        echoes = record.get("echoes", [])
        wrote = [e for e in echoes if e.startswith("wrote ")]
        # The successful file's pair + the partial pair's .md still print.
        assert len(wrote) == 3, echoes
        # But no "complete" line — one file failed.
        assert all("complete" not in e for e in echoes), echoes

    def test_success_full_pairs_prints_complete(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """Plain path, all files fully written, exit 0 -> exactly one
        "complete" line, after the wrote lines."""
        files = touch_files(["a.m4a"], tmp_path)
        code = _run_preset_plain(
            tmp_path,
            monkeypatch,
            files,
            result_per_file=_good_result(_two_label_paragraphs()),
        )
        assert code == 0
        out = capsys.readouterr().out
        assert out.count("complete") == 1
        wrote_idx = out.index("wrote ")
        complete_idx = out.index("complete")
        assert complete_idx > wrote_idx

    def test_quiet_failed_run_prints_no_complete(self, tmp_path, monkeypatch):
        """--quiet with a failed run: nothing at all (no wrote lines, no
        complete line)."""
        files = touch_files(["a.m4a"], tmp_path)
        record: dict[str, Any] = {}

        def fake_echo(msg, err: bool = False, **kw):
            if not err:
                record.setdefault("echoes", []).append(str(msg))

        def bad_transcribe(file, options=None, glossary_path=None, **kwargs):
            return {"text": "", "segments": []}

        _plain_seams(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        tracker = begin_interrupt_tracking()
        with (
            mock.patch.object(
                batch_preset_module, "_transcribe_preset_file", bad_transcribe
            ),
            mock.patch.object(batch_preset_module.typer, "echo", fake_echo),
        ):
            code = batch_preset_module.run_preset(
                list(files),
                command="meeting",
                config_path=None,
                glossary_path=None,
                quiet=True,
                yes=True,
                tracker=tracker,
                display=None,
            )
        assert code == 1
        assert record.get("echoes", []) == []

    def test_group_partial_pair_no_complete_line(self, tmp_path, monkeypatch):
        """Group path: a group's .json write fails (partial pair) -> exit
        1, the .md "wrote" line prints, but NO "complete" line."""
        files = touch_files(["a.m4a", "b.m4a"], tmp_path)
        record: dict[str, Any] = {}

        def fake_echo(msg, err: bool = False, **kw):
            if not err:
                record.setdefault("echoes", []).append(str(msg))

        _patch_group_seams(monkeypatch, tmp_path)
        tracker = begin_interrupt_tracking()
        with (
            mock.patch.object(
                batch_preset_module, "_write_preset_output", _group_write_seam
            ),
            mock.patch.object(batch_preset_module.typer, "echo", fake_echo),
        ):
            code = batch_preset_module.run_preset(
                list(files),
                command="meeting",
                config_path=None,
                glossary_path=None,
                quiet=False,
                yes=True,
                tracker=tracker,
                display=None,
            )
        assert code == 1
        echoes = record.get("echoes", [])
        assert any(e.startswith("wrote ") for e in echoes), echoes
        assert all("complete" not in e for e in echoes), echoes

    def test_group_single_pair_success_prints_complete(self, tmp_path, monkeypatch):
        """Group path: 3 files yield 3 groups (the zero-byte fixtures force
        a break at every boundary) and every group writes a full pair ->
        exit 0, the final "complete" line DOES print. The old gate
        (``len(files) * len(PRESET_FORMATS)`` = 6) would still pass here,
        so this pins the plain-per-group arithmetic; the merged-group case
        (2 files -> 1 pair, old gate expected 4, actual 2) is the real
        regression this fix addresses and is asserted in the unit test
        below."""
        files = touch_files(["a.m4a", "b.m4a", "c.m4a"], tmp_path)
        record: dict[str, Any] = {}

        def fake_echo(msg, err: bool = False, **kw):
            if not err:
                record.setdefault("echoes", []).append(str(msg))

        def full_write_seam(result, first_stem, out_dir, *, date_str=None):
            # A full pair (md + json) for each group.
            return [f"2025-01-01 {first_stem}.md", f"2025-01-01 {first_stem}.json"]

        _patch_group_seams(monkeypatch, tmp_path)
        tracker = begin_interrupt_tracking()
        with (
            mock.patch.object(
                batch_preset_module, "_write_preset_output", full_write_seam
            ),
            mock.patch.object(batch_preset_module.typer, "echo", fake_echo),
        ):
            code = batch_preset_module.run_preset(
                list(files),
                command="meeting",
                config_path=None,
                glossary_path=None,
                quiet=False,
                yes=True,
                tracker=tracker,
                display=None,
            )
        assert code == 0
        echoes = record.get("echoes", [])
        # 3 groups -> 3 full pairs -> 6 "wrote" lines.
        wrote = [e for e in echoes if e.startswith("wrote ")]
        assert len(wrote) == 6, echoes
        # The final line prints (the gate counted groups, not files).
        complete_lines = [e for e in echoes if "complete" in e]
        assert len(complete_lines) == 1, echoes
        assert complete_lines[0] == "\u2713 complete — wrote 6 file(s)", echoes


def test_run_went_full_group_gate_unit() -> None:
    """The group-path gate counts groups (pairs written), not files: the
    real regression case is 2 files merged into ONE group -> one pair (2
    files written), where the old ``len(files) * len(PRESET_FORMATS)`` = 4
    gate wrongly failed (2 != 4) and suppressed the line. The fixed gate
    (``expected_pairs`` = written // formats = 1) passes."""
    # 2 files, 1 merged group -> one full pair written (2 files).
    written_2files_1group = ["a.md", "a.json"]
    assert run_went_full(written_2files_1group, 1, 0) is True
    # The old (buggy) arithmetic would have failed this:
    assert 2 * len(PRESET_FORMATS) != 2
    # A mixed batch (1 merged group full + 1 standalone partial) fails the
    # gate: 3 groups, 1 partial pair -> 5 files != 3 * 2 = 6.
    written_mixed = ["a.md", "a.json", "b.md", "c.md", "c.json"]
    assert run_went_full(written_mixed, 3, 1) is False
