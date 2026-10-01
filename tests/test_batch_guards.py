"""Per-file decode guards for the batch loops (issue #77 merge gate).

Every decode loop (``transcribe_batch``, the plain loop and both group
branches of ``run_batch``, and ``run_preset`` for meeting/memo) must
degrade an unexpected ``transcribe_file`` exception to a clean
``error:`` line (exit 1, run continues) through one shared guard —
``_guarded_transcribe`` in :mod:`vemoizer.batch_output`, wrapped by
``_transcribe_guarded`` in :mod:`vemoizer.batch`.

All tests mock ``transcribe_file``; no models, no network, no ffmpeg.
"""

from __future__ import annotations

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

import vemoizer.batch as batch
import vemoizer.batch_output as batch_output
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()


def _options() -> batch.RunOptions:
    return batch.RunOptions.expert_transcribe(
        profile="dictation",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )


def _touch(files) -> None:
    for f in files:
        f.touch()


def test_run_batch_all_single_part_groups_middle_failure_is_clean(
    tmp_path, monkeypatch, capsys
) -> None:
    """A grouped run of 3 files where EVERY boundary is a break (three
    single-file groups): the fake ``transcribe_file`` raises
    ``RuntimeError`` for the middle file -> exit 1, exactly one clean
    ``error: b.m4a: ...`` line, outputs for files 1 and 3 written,
    nothing for file 2, and no traceback escapes."""
    import vemoizer.grouping as grouping

    decoded: list[str] = []

    def fake_transcribe_file(path, **kwargs):
        decoded.append(path.name)
        if path.name == "b.m4a":
            raise RuntimeError("decoder exploded")
        return {"text": "hei", "segments": []}

    def fake_decode_boundaries(files, transcribe_fn=None):
        # Both boundaries break (empty tail/head texts) -> three
        # single-file groups, all taken through the single-part branch.
        return ["", ""], ["", ""]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(batch, "_resolve_llm_config", lambda p: None)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    files = [tmp_path / "a.m4a", tmp_path / "b.m4a", tmp_path / "c.m4a"]
    _touch(files)
    code = batch.run_batch(files, _options(), formats=["txt"], yes=True)
    assert code == 1
    err = capsys.readouterr().err
    assert "error: b.m4a: decoder exploded" in err
    assert err.count("error:") == 1
    assert "Traceback" not in err
    # All three groups were attempted; files 1 and 3 written, file 2 not.
    assert decoded == ["a.m4a", "b.m4a", "c.m4a"]
    assert (tmp_path / "a.txt").exists()
    assert (tmp_path / "c.txt").exists()
    assert not (tmp_path / "b.txt").exists()
    assert not (tmp_path / "b.json").exists()


def test_run_batch_single_part_group_keyboard_interrupt_still_propagates(
    tmp_path, monkeypatch
) -> None:
    """A ``KeyboardInterrupt`` from ``transcribe_file`` in a single-part
    group must still propagate — not be swallowed into an error line."""
    import vemoizer.grouping as grouping

    def fake_transcribe_file(path, **kwargs):
        raise KeyboardInterrupt()

    def fake_decode_boundaries(files, transcribe_fn=None):
        return ["", ""], ["", ""]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(batch, "_resolve_llm_config", lambda p: None)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    files = [tmp_path / "a.m4a", tmp_path / "b.m4a", tmp_path / "c.m4a"]
    _touch(files)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(files, _options(), formats=["txt"], yes=True)


# -- run_preset (meeting/memo) guard (issue #77 merge gate) ----------------
#
# run_preset has its own inline copy of the per-file transcribe core
# (transcribe_file with its own temp glossary argument); it must carry
# the identical per-file contract through the SAME shared guard
# (batch_output._guarded_transcribe) — not a fourth handler copy.


def _run_preset_guard_case(tmp_path, monkeypatch, command: str) -> None:
    """Drive the *command* preset over a.m4a/b.m4a/c.m4a where the fake
    ``transcribe_file`` raises ``RuntimeError`` for the middle file.

    Pinned contract: exit 1, exactly one clean ``error: b.m4a: ...``
    line on the runner's captured stderr (the CliRunner captures typer's
    ``err=True`` output itself, so ``result.stderr`` — not capsys — is
    the right surface, same as tests/test_meeting_memo_cli.py), no
    traceback, outputs for files 1 and 3 written, nothing for file 2,
    and the temp glossary file deleted.
    """

    def fake_transcribe_file(path, **kwargs):
        if path.name == "b.m4a":
            raise RuntimeError("decoder exploded")
        return {"text": "hei", "segments": [], "notes": {"title": path.stem.upper()}}

    # Seam: run_preset binds the FUNCTION OBJECT from the pipeline module
    # (deferred `from vemoizer.pipeline import transcribe_file`), so the
    # pipeline module attribute is patched in addition to the batch-layer
    # binding so every resolution path sees the fake.
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    # A per-file title in the result makes the .md/.json pair names
    # distinct (the preset derives the base name from notes["title"]
    # with a fallback to the first file's stem).
    _touch([tmp_path / n for n in ("a.m4a", "b.m4a", "c.m4a")])
    result = runner.invoke(app, [command, "a.m4a", "b.m4a", "c.m4a"])
    assert result.exit_code == 1
    err = result.stderr
    assert err.count("error:") == 1
    assert "Traceback" not in err
    md = sorted(p.name for p in tmp_path.glob("*.md"))
    js = sorted(p.name for p in tmp_path.glob("*.json"))
    # Per-file title -> per-file dated pair (issue #82: base name is
    # YYYY-MM-DD <title>; the date is today's, so we check only the stem).
    assert len(md) == 2 and "A" in md[0] and "C" in md[1], md
    assert len(js) == 2 and "A" in js[0] and "C" in js[1], js
    assert "B" not in " ".join(md + js)
    # The composed-glossary temp file is deleted after the run (finally).
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("vemoizer-")]


def test_run_preset_meeting_middle_failure_is_clean(tmp_path, monkeypatch) -> None:
    """meeting (via run_preset), 3 files, middle raises RuntimeError:
    exit 1, one clean error line, outputs for files 1 and 3 only, temp
    glossary file gone."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "glossary.txt").write_text("Nordea\n", encoding="utf-8")
    files = [tmp_path / n for n in ("a.m4a", "b.m4a", "c.m4a")]
    _touch(files)
    _run_preset_guard_case(tmp_path, monkeypatch, "meeting")


def test_run_preset_memo_middle_failure_is_clean(tmp_path, monkeypatch) -> None:
    """memo (via run_preset): identical per-file contract."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "glossary.txt").write_text(
        "Blacksit => Flagship\n", encoding="utf-8"
    )
    files = [tmp_path / n for n in ("a.m4a", "b.m4a", "c.m4a")]
    _touch(files)
    _run_preset_guard_case(tmp_path, monkeypatch, "memo")


def test_run_preset_keyboard_interrupt_propagates_and_glossary_cleaned(
    tmp_path, monkeypatch
) -> None:
    """A ``KeyboardInterrupt`` from ``transcribe_file`` must still
    propagate out of run_preset (not be swallowed into an error line)
    AND the temp glossary file must still be deleted (the finally).

    A glossary layer is written so the temp file IS created (the
    finally-cleanup path is exercised, not the no-layer trivial case).
    """
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "glossary.txt").write_text("Nordea\n", encoding="utf-8")

    def fake_transcribe_file(path, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        batch_output.run_preset(
            [tmp_path / "a.m4a"],
            command="meeting",
            config_path=None,
            glossary_path=None,
        )
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("vemoizer-")]
