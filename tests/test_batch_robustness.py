"""M3 robustness fixes from the lens review of PR #86 (issue #77).

Four HIGH findings:

1. ``batch_output._check_result``: a non-list ``warnings`` key (e.g. a
   lone string) used to crash with a ``TypeError``; a lone string is
   now treated as a one-element list (a list/tuple of strings still
   echoes each, anything else is no warnings).
2. ``run_batch``: when the default boundary transcriber has to
   download/load its model and that fails with an exception type
   ``_edge_text`` does not catch (``IngestError``/``RuntimeError``/
   ``OSError``), the raw traceback used to escape — now a clean
   ``error: could not decode boundaries: ...`` line and exit 1, with
   nothing decoded or written afterwards. ``KeyboardInterrupt`` still
   propagates.
3. The transcribe -> check -> write core is ONE shared helper
   (``batch._process_result``); the per-file and per-group loops are
   thin adapters (pinned by the existing test suite).
4. ``ingest._DrainState``: the reader-thread handoff is a typed
   dataclass instead of a mixed ``list[object]`` (behaviour unchanged).

All models mocked; no network, no model downloads, no ffmpeg beyond the
fake-Popen seams already used in ``test_ingest.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

import vemoizer.batch as batch
import vemoizer.batch_output as batch_output
import vemoizer.ingest as ingest
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


def _touch(files: list[Path]) -> None:
    for f in files:
        f.touch()


# ---------------------------------------------------------------------------
# Finding 1 — _check_result: non-list ``warnings`` must not crash
# ---------------------------------------------------------------------------


def test_check_result_lone_string_warning_is_echoed_not_crash(tmp_path, capsys) -> None:
    """A ``warnings`` key holding a lone string (not a list) is treated
    as a one-element list: echoed, no TypeError (issue #77 lens finding)."""
    result = {"text": "x", "segments": [{"text": "a"}], "warnings": "not-a-list"}
    code = batch_output._check_result(Path("x.m4a"), result, diarize=False)
    assert code == 0
    err = capsys.readouterr().err
    assert "not-a-list" in err


def test_check_result_tuple_of_strings_is_echoed(tmp_path, capsys) -> None:
    """A tuple of strings is a valid warnings channel (list/tuple contract)."""
    result = {"text": "x", "segments": [], "warnings": ("w1", "w2")}
    code = batch_output._check_result(Path("x.m4a"), result, diarize=False)
    assert code == 0
    err = capsys.readouterr().err
    assert "w1" in err
    assert "w2" in err


def test_check_result_non_iterable_junk_warning_is_ignored(tmp_path, capsys) -> None:
    """Anything else (int, dict, ...) is dropped without a crash and the
    result is still valid — but the degradation is observable: exactly
    one bounded stderr line names the unparseable payload's type. An
    absent key or an explicit ``None`` stays silent (no warnings at
    all, nothing to report)."""
    result = {"text": "x", "segments": [{"text": "a"}], "warnings": 42}
    code = batch_output._check_result(Path("x.m4a"), result, diarize=False)
    assert code == 0
    err = capsys.readouterr().err
    assert err == "warning: ignored an unparseable warnings payload of type int\n"

    # No warnings key at all: still silent (no warnings, no diagnostic).
    result = {"text": "x", "segments": [{"text": "a"}]}
    code = batch_output._check_result(Path("x.m4a"), result, diarize=False)
    assert code == 0
    assert capsys.readouterr().err == ""

    # A lone dict is unparseable too — same one-line diagnostic.
    result = {"text": "x", "segments": [{"text": "a"}], "warnings": {"a": 1}}
    code = batch_output._check_result(Path("x.m4a"), result, diarize=False)
    assert code == 0
    err = capsys.readouterr().err
    assert err == "warning: ignored an unparseable warnings payload of type dict\n"


# ---------------------------------------------------------------------------
# Finding 2 — boundary model load failure must not escape as a traceback
# ---------------------------------------------------------------------------


def _fake_edge_window(path, start, end) -> np.ndarray:
    """Non-empty 1 s window; the real transcribe_fn is never reached."""
    return np.zeros(16_000, dtype=np.float32)


class _HfDownloadError(Exception):
    """Mimics a huggingface_hub download failure (not OSError/
    RuntimeError/IngestError — the types _edge_text does not catch)."""


def test_run_batch_boundary_transcriber_load_failure_is_clean(
    tmp_path, monkeypatch, capsys
) -> None:
    """2 files, --yes: the boundary decodes run with the DEFAULT
    transcriber; if loading it raises a non-``OSError`` exception type
    (e.g. a HF download failure), run_batch reports a clean
    ``error: could not decode boundaries: ...`` line and exits 1 — no
    traceback, nothing decoded, nothing written."""
    import vemoizer.whisper_transcriber as wt

    decode_calls: list[str] = []

    def fake_transcribe_file(path, **kwargs):
        decode_calls.append(str(path))
        return {"text": "hei", "segments": []}

    def boom(language=None):
        raise _HfDownloadError("HF download failed")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(wt, "WhisperTranscriber", boom)
    import vemoizer.grouping as grouping

    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 60.0)
    monkeypatch.setattr(grouping, "_decode_edge_window", _fake_edge_window)
    monkeypatch.chdir(tmp_path)

    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    _touch(files)
    code = batch.run_batch(files, _options(), formats=["txt"], yes=True)

    err = capsys.readouterr().err
    assert code == 1
    assert "could not decode boundaries" in err
    assert "HF download failed" in err
    assert "Traceback" not in err
    # Nothing decoded, nothing written afterwards.
    assert decode_calls == []
    assert (tmp_path / "Uusi äänitys 425.txt").exists() is False


def test_run_batch_boundary_transcriber_failure_cli_level(
    tmp_path, monkeypatch
) -> None:
    """Same failure at the CLI level (CliRunner): exit code 1, clean
    message, nothing written."""
    import vemoizer.whisper_transcriber as wt

    def boom(language=None):
        raise _HfDownloadError("HF download failed")

    monkeypatch.setattr(wt, "WhisperTranscriber", boom)
    import vemoizer.grouping as grouping

    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 60.0)
    monkeypatch.setattr(grouping, "_decode_edge_window", _fake_edge_window)
    monkeypatch.chdir(tmp_path)

    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    _touch(files)
    result = runner.invoke(
        app,
        ["transcribe", "Uusi äänitys 425.m4a", "Uusi äänitys 426.m4a", "--yes"],
    )
    assert result.exit_code == 1
    assert "could not decode boundaries" in result.stderr
    assert "Traceback" not in result.stderr
    # Nothing decoded or written.
    assert (tmp_path / "Uusi äänitys 425.txt").exists() is False
    assert (tmp_path / "Uusi äänitys 426.txt").exists() is False


def test_run_batch_boundary_load_keyboard_interrupt_still_propagates(
    tmp_path, monkeypatch
) -> None:
    """KeyboardInterrupt from the boundary transcriber load must still
    propagate (not be swallowed into an error line)."""
    import vemoizer.whisper_transcriber as wt

    def boom(language=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(wt, "WhisperTranscriber", boom)
    import vemoizer.grouping as grouping

    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 60.0)
    monkeypatch.setattr(grouping, "_decode_edge_window", _fake_edge_window)
    monkeypatch.chdir(tmp_path)

    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    _touch(files)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(files, _options(), formats=["txt"], yes=True)


# ---------------------------------------------------------------------------
# Finding 4 — _drain_ffmpeg_pcm: typed drain state, unchanged behaviour
# ---------------------------------------------------------------------------


def test_drain_state_defaults() -> None:
    state = ingest._DrainState()
    assert state.total == 0
    assert state.error is None
    state.total = 1234
    assert state.total == 1234


class _StderrFile:
    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def read(self, size: int | None = None) -> bytes:
        return b"some stderr here"


class _ProcStub:
    """A Popen-shaped stub for driving _drain_ffmpeg_pcm directly."""

    returncode = 0

    def __init__(self, pipe) -> None:
        self.stdout = pipe
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return self.returncode

    def kill(self) -> None:
        self.killed = True


class _BrokenPipe:
    def read(self, size: int | None = None) -> bytes:
        raise OSError("broken pipe")


def test_drain_state_reader_exception_stored_and_propagated() -> None:
    """A reader-side exception (e.g. broken pipe) is stored on the state
    and re-raised by the drain — no silent thread death."""
    proc = _ProcStub(_BrokenPipe())
    with pytest.raises(OSError, match="broken pipe"):
        ingest._drain_ffmpeg_pcm(proc, _StderrFile(), timeout=0.2)
    # The process was reaped (already exited: no kill needed).
    assert proc.waited


def test_drain_state_normal_completion_returns_byte_count() -> None:
    """The happy path: the reader counts bytes, the process exits 0, and
    the drain returns (total bytes, bounded stderr excerpt)."""
    raw = np.zeros(16_000, dtype=np.float32).tobytes()
    pos = 0

    class _Pipe:
        def read(self, size: int | None = None) -> bytes:
            nonlocal pos
            chunk = raw[pos : pos + (size or len(raw))]
            pos += len(chunk)
            return chunk

    proc = _ProcStub(_Pipe())
    total, stderr = ingest._drain_ffmpeg_pcm(proc, _StderrFile(), timeout=5.0)
    assert total == len(raw)
    assert stderr == "some stderr here"
    assert proc.waited
    assert not proc.killed


def test_pcm_duration_reader_pipe_error_is_clean_ingest_error(tmp_path) -> None:
    """A reader-side pipe error during pcm_duration_seconds is a clean
    IngestError (not a raw OSError traceback)."""

    class _ProcNoWait:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = _BrokenPipe()
            self.killed = False
            self.waited = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    f = tmp_path / "x.m4a"
    f.touch()
    with (
        patch("vemoizer.ingest.subprocess.Popen", return_value=_ProcNoWait()),
        pytest.raises(ingest.IngestError),
    ):
        ingest.pcm_duration_seconds(f, timeout=0.2)


# ---------------------------------------------------------------------------
# Finding 3 — the shared per-file/per-group core is used by both loops
# ---------------------------------------------------------------------------


def test_process_result_writes_all_formats_and_echoes(
    tmp_path, monkeypatch, capsys
) -> None:
    """``_process_result``: a good result writes every format from the
    label's stem and prints the quiet-suppressed echo line."""
    monkeypatch.chdir(tmp_path)
    label = "Uusi äänitys 425.m4a"
    result = {"text": "hei", "segments": []}
    ok = batch._process_result(
        label, result, formats=["txt", "json"], out=None, quiet=False, options=None
    )
    assert ok
    assert (tmp_path / "Uusi äänitys 425.txt").exists()
    assert (tmp_path / "Uusi äänitys 425.json").exists()
    assert "wrote transcript for Uusi äänitys 425.m4a" in capsys.readouterr().out


def test_process_result_out_overrides_and_first_format_only(
    tmp_path, monkeypatch, capsys
) -> None:
    """``--out`` target: only the first format is written to it."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "merged.txt"
    result = {"text": "hei", "segments": []}
    ok = batch._process_result(
        "x.m4a",
        result,
        formats=["txt", "json"],
        out=out,
        quiet=False,
        options=None,
    )
    assert ok
    assert out.exists()
    assert not (tmp_path / "x.txt").exists()


def test_process_result_stdout_target(tmp_path, monkeypatch, capsys) -> None:
    """``--out -``: the transcript streams to stdout."""
    monkeypatch.chdir(tmp_path)
    result = {"text": "hei maailma", "segments": []}
    ok = batch._process_result(
        "x.m4a", result, formats=["txt"], out=Path("-"), quiet=False, options=None
    )
    assert ok
    assert "hei maailma" in capsys.readouterr().out


def test_process_result_copy_is_batch_only(tmp_path, monkeypatch) -> None:
    """``--copy`` is honored only by transcribe_batch (options=None): the
    clipboard seam is called there and not on the group path."""
    import vemoizer.copy as copy_mod

    copied: list[str] = []
    monkeypatch.setattr(copy_mod, "copy_to_clipboard", lambda t: copied.append(t))
    monkeypatch.chdir(tmp_path)
    result = {"text": "hei", "segments": []}
    options = _options()
    # Group path: no copy.
    ok = batch._process_result(
        "x.m4a", result, formats=["txt"], out=None, quiet=True, options=options
    )
    assert ok
    assert copied == []
    # Batch path (options=None): copy.
    ok = batch._process_result(
        "x.m4a", result, formats=["txt"], out=None, quiet=True, options=None
    )
    assert ok
    assert copied == ["hei"]
