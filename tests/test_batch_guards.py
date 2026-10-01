"""Single-part group branch guards for ``run_batch`` (issue #77 merge gate).

The plain loop and the multi-part group branch of ``run_batch`` already
degrade an unexpected ``transcribe_file`` exception to a clean
``error:`` line (exit 1, batch continues). The single-part group branch
(the common case in a multi-file grouped run: most groups are one file)
must carry the identical contract.

All tests mock ``transcribe_file``; no models, no network, no ffmpeg.
"""

from __future__ import annotations

import pytest

import vemoizer.batch as batch
import vemoizer.pipeline as pipeline_module


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
