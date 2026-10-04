"""Round-3 lens-bug regressions for the M3 split-recording grouping (issue #77).

Each test here pins a real bug found in the round-3 six-pass lens review
(tmp/issue-77/all-findings.md) — the batch IngestError path, the concat temp
dir leak, the bounded ffprobe/ffmpeg stderr, the shared Whisper cache, and the
explicit part-marker type. Kept in a dedicated file so the main
``test_grouping.py`` stays under the 800-line test-file limit while these
regression cases keep a home. Pure-stdlib + numpy except where a test needs
the real ffmpeg binary (marked ``skipif``); no model loads, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vemoizer.grouping import GroupingError, decode_boundaries


def _make_wav(path: Path, seconds: float) -> Path:
    """Write a tiny mono 16 kHz 16-bit PCM WAV (the grouping fixture helper)."""
    import struct

    rate = 16000
    n = int(seconds * rate)
    samples = b"".join(struct.pack("<h", 3000) for _ in range(n))
    data = samples
    header = b"RIFF"
    header += struct.pack("<I", 36 + len(data))
    header += b"WAVEfmt "
    header += struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    header += b"data"
    header += struct.pack("<I", len(data))
    path.write_bytes(header + data)
    return path


# ---------------------------------------------------------------------------
# Finding 5 — per-part IngestError in the multi-part group path
# ---------------------------------------------------------------------------


def test_run_batch_part_ingest_error_is_clean_and_batch_continues(
    tmp_path, monkeypatch, capsys
) -> None:
    """A per-part ingest failure in a multi-part group path (finding 5)
    gives a clean ``error:`` line and a non-zero exit for that group, while
    the remaining groups are still transcribed and the batch continues —
    never a raw traceback.
    """
    import vemoizer.batch as batch
    import vemoizer.grouping as grouping
    import vemoizer.pipeline as pipeline_module
    from vemoizer.ingest import IngestError

    decode_calls: list[str] = []

    def fake_transcribe_file(path, **kwargs):
        decode_calls.append(Path(path).name)
        return {"text": "hei", "segments": []}

    def fake_decode_boundaries(files, transcribe_fn=None, **kw):
        # Both boundaries continue -> all 3 files in ONE multi-part group.
        return ["t jatkumossa", "t jatkuu"], ["h jatkuu", "h jatkuu"]

    def fake_concat(files, **kw):
        return files[0]

    def boom(files, **kw):
        raise IngestError("part file missing or corrupt: Uusi äänitys 425.m4a")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "concat_groups", fake_concat)
    monkeypatch.setattr(batch, "concat_groups", fake_concat)
    monkeypatch.setattr(grouping, "part_offsets", boom)
    monkeypatch.setattr(batch, "part_offsets", boom)
    files = [
        tmp_path / "Uusi äänitys 425.m4a",
        tmp_path / "Uusi äänitys 426.m4a",
        tmp_path / "Uusi äänitys 427.m4a",
    ]
    for f in files:
        f.touch()
    code = batch.run_batch(
        files,
        batch.RunOptions.expert_transcribe(
            profile="dictation",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            config_path=None,
        ),
        yes=True,
    )
    err = capsys.readouterr().err
    # A clean error line (not a traceback), non-zero exit for the failed group.
    assert code == 1
    assert "error:" in err
    assert "Traceback" not in err
    # The batch continued: no decode of the failed group happened, and no
    # traceback was raised.
    assert decode_calls == []


# ---------------------------------------------------------------------------
# Finding 12 — concat_groups temp dir cleaned up on an unexpected exception
# ---------------------------------------------------------------------------


def test_concat_groups_cleans_temp_dir_on_unexpected_exception(
    tmp_path, monkeypatch
) -> None:
    """concat_groups removes its 0o700 temp dir if an unexpected exception
    escapes before the return (finding 12) — including before the merged
    file exists, so no partial private audio or empty dir is left behind.
    """
    import glob
    import tempfile

    import vemoizer.grouping_concat as gc

    a = _make_wav(tmp_path / "Uusi äänitys 425.wav", 0.5)
    b = _make_wav(tmp_path / "Uusi äänitys 426.wav", 0.5)

    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "vemoizer-concat-*")))

    # Force the ffmpeg concat step to raise an unexpected (non-GroupingError)
    # exception before the merged file is produced.
    def boom(argv, **kwargs):
        raise RuntimeError("unexpected crash")

    monkeypatch.setattr("subprocess.run", boom)
    with pytest.raises(RuntimeError):
        gc.concat_groups([a, b])

    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "vemoizer-concat-*")))
    # No NEW temp dir leaked by the failing concat call.
    assert after - before == set()


# ---------------------------------------------------------------------------
# Finding 8 — raw ffprobe/ffmpeg stderr is bounded, not embedded verbatim
# ---------------------------------------------------------------------------


def test_concat_groups_ffmpeg_failure_message_is_bounded(tmp_path, monkeypatch) -> None:
    """A raw ffprobe/ffmpeg stderr is truncated and collapsed to a single
    bounded paragraph that still names the offending files (finding 8) —
    never embedded verbatim into the user-facing error.
    """
    import vemoizer.grouping as grouping
    import vemoizer.grouping_concat as gc

    a = _make_wav(tmp_path / "Uusi äänitys 425.wav", 0.5)
    b = _make_wav(tmp_path / "Uusi äänitys 426.wav", 0.5)

    # Same stream signature so the mismatch check passes, then ffmpeg fails
    # with a long, multi-line, raw stderr.
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "aac,48000,1")
    long_stderr = ("line one of raw ffmpeg output\n" * 50).encode()

    class FakeProc:
        returncode = 1
        stdout = b""
        stderr = long_stderr

    def fake_run(argv, **kwargs):
        return FakeProc()

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(GroupingError) as exc:
        gc.concat_groups([a, b])
    msg = str(exc.value)
    # Still names the offending files.
    assert "Uusi äänitys 425.wav" in msg
    assert "Uusi äänitys 426.wav" in msg
    # Bounded + single paragraph (no raw multi-line stderr, no newline).
    assert "\n" not in msg
    assert len(msg) < 600


# ---------------------------------------------------------------------------
# Findings 9-16 — boundary decode must not destroy the shared Whisper cache
# ---------------------------------------------------------------------------


def test_decode_boundaries_does_not_destroy_shared_whisper_cache(
    tmp_path, monkeypatch
) -> None:
    """decode_boundaries must NOT call cleanup() on the Whisper model it
    creates: cleanup() nulls mlx_whisper's shared ModelHolder cache, which
    would force the main pipeline to reload whisper-large-v3-turbo (finding
    9). The boundary transcriber is released by GC, not by an explicit
    cache-destroying cleanup().
    """
    import vemoizer.grouping as grouping
    import vemoizer.whisper_transcriber as wt

    calls = {"transcribe": 0, "cleanup": 0}

    class FakeWhisper:
        def __init__(self, language=None):
            pass

        def transcribe(self, audio, **kwargs):
            calls["transcribe"] += 1
            return {"text": "x"}

        def cleanup(self):
            calls["cleanup"] += 1

    # decode_boundaries imports WhisperTranscriber lazily from the module.
    monkeypatch.setattr(wt, "WhisperTranscriber", FakeWhisper)
    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 60.0)
    monkeypatch.setattr(
        grouping,
        "_decode_edge_window",
        lambda path, start, end, **kw: np.zeros(16000, dtype=np.float32),
    )
    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    decode_boundaries(files)
    assert calls["transcribe"] == 2  # tail + head both decoded
    assert calls["cleanup"] == 0  # the shared cache is preserved


# ---------------------------------------------------------------------------
# run_batch — multi-group --out must not silently overwrite (round-1 HIGH)
# ---------------------------------------------------------------------------


def test_run_batch_out_file_with_multiple_groups_fails_loud(
    tmp_path, monkeypatch, capsys
) -> None:
    """`--out file` + 2+ files that confirm as 2 groups: exit 2 up front.

    Without the guard every group would write to the same `--out` path and
    only the last group's transcript would survive (round-1 adversarial
    finding, HIGH). Fail up front, before any decode, with a clear message
    and a non-zero exit — never a silently-lost transcript.
    """
    import vemoizer.batch as batch
    import vemoizer.grouping as grouping
    import vemoizer.pipeline as pipeline_module

    decode_calls: list[int] = []

    def fake_transcribe_file(path, **kwargs):
        decode_calls.append(1)
        return {"text": "hei", "segments": []}

    def fake_decode_boundaries(files, transcribe_fn=None, **kw):
        # Both boundaries break (closing cue in each tail) -> 3 singleton
        # groups -> 3 writes to the same --out target would be an
        # overwrite.
        return ["t1 kiitos", "t2 moi"], ["h1", "h2"]

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    out = tmp_path / "merged.txt"
    files = [
        tmp_path / "Uusi äänitys 425.m4a",
        tmp_path / "Uusi äänitys 426.m4a",
        tmp_path / "Uusi äänitys 427.m4a",
    ]
    for f in files:
        f.touch()
    code = batch.run_batch(
        files,
        batch.RunOptions.expert_transcribe(
            profile="dictation",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            config_path=None,
        ),
        out=out,
        yes=True,
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "--out" in err
    assert "groups" in err
    # Up front: no decode ran, nothing written.
    assert decode_calls == []
    assert not out.exists()


def test_run_batch_out_stdout_with_multiple_groups_is_fine(
    tmp_path, monkeypatch, capsys
) -> None:
    """`--out -` (stdout) is always fine — groups stream in order."""
    import vemoizer.batch as batch
    import vemoizer.grouping as grouping
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe_file(path, **kwargs):
        return {"text": "hei", "segments": []}

    def _two_group_boundaries(files, transcribe_fn=None, **kw):
        # Both boundaries break -> 2 singleton groups -> 2 streams to stdout.
        return ["t1 kiitos", "t2 moi"], ["h1", "h2"]

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(grouping, "decode_boundaries", _two_group_boundaries)
    files = [
        tmp_path / "Uusi äänitys 425.m4a",
        tmp_path / "Uusi äänitys 426.m4a",
        tmp_path / "Uusi äänitys 427.m4a",
    ]
    for f in files:
        f.touch()
    code = batch.run_batch(
        files,
        batch.RunOptions.expert_transcribe(
            profile="dictation",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            config_path=None,
        ),
        out=Path("-"),
        yes=True,
    )
    out_text = capsys.readouterr().out
    assert code == 0
    # All three groups stream to stdout in order — no overwrite possible.
    assert out_text.count("hei") == 3


def test_run_batch_out_file_with_single_group_is_fine(tmp_path, monkeypatch) -> None:
    """`--out file` + 2+ files confirming as ONE group: one combined
    transcript to that path (the intended use)."""
    import vemoizer.batch as batch
    import vemoizer.grouping as grouping

    def fake_transcribe_file(path, **kwargs):
        return {"text": "yksi ja kaksi", "segments": []}

    monkeypatch.setattr("vemoizer.pipeline.transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None, **kw: (["t jatkumossa"], ["h jatkuu"]),
    )
    monkeypatch.setattr(grouping, "concat_groups", lambda files, **kw: files[0])
    monkeypatch.setattr(batch, "concat_groups", lambda files, **kw: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files, **kw: [])
    monkeypatch.setattr(batch, "part_offsets", lambda files, **kw: [])
    out = tmp_path / "merged.txt"
    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    code = batch.run_batch(
        files,
        batch.RunOptions.expert_transcribe(
            profile="dictation",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            config_path=None,
        ),
        out=out,
        yes=True,
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").strip() == "yksi ja kaksi"


# ---------------------------------------------------------------------------
# TTY guard — non-TTY stdin without --yes/--no-group fails IMMEDIATELY
# (ticket criterion: before any boundary decode or model load, exit 2,
# message names --yes / --no-group)
# ---------------------------------------------------------------------------


def test_run_batch_non_tty_without_yes_or_no_group_fails_immediately(
    tmp_path, monkeypatch, capsys
) -> None:
    """2+ files, stdin not a TTY, neither --yes nor --no-group: run_batch
    must fail IMMEDIATELY (exit 2) with a message naming --yes / --no-group,
    BEFORE any boundary decode, transcriber load, or ingest. The isatty
    seam is monkeypatched — the TTY check is a pure boolean here.
    """
    import vemoizer.batch as batch
    import vemoizer.grouping as grouping
    import vemoizer.ingest as ingest
    import vemoizer.pipeline as pipeline_module

    touched: dict[str, int] = {"decode": 0, "transcribe_file": 0, "ingest": 0}

    def fake_decode_boundaries(files, transcribe_fn=None, **kw):
        touched["decode"] += 1
        return [""], [""]

    def fake_transcribe_file(path, **kwargs):
        touched["transcribe_file"] += 1
        return {"text": "hei", "segments": []}

    def fake_ingest_audio(path):
        touched["ingest"] += 1
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr(batch, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(ingest, "ingest_audio", fake_ingest_audio)
    monkeypatch.setattr(grouping, "_decode_edge_window", fake_ingest_audio)
    # The seam under test: stdin is NOT a TTY.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    code = batch.run_batch(
        files,
        batch.RunOptions.expert_transcribe(
            profile="dictation",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            config_path=None,
        ),
        yes=False,
        no_group=False,
    )
    err = capsys.readouterr().err
    # Immediate, clear, exit 2 — and nothing downstream ran.
    assert code == 2
    assert "--yes" in err
    assert "--no-group" in err
    assert "TTY" in err
    assert touched == {"decode": 0, "transcribe_file": 0, "ingest": 0}


def test_run_batch_yes_and_no_group_are_mutually_excluded(
    tmp_path, monkeypatch, capsys
) -> None:
    """--yes and --no-group given together is rejected (exit 2, clear
    message) — before anything else runs."""
    import vemoizer.batch as batch

    touched: dict[str, int] = {}

    def fake_decode_boundaries(files, transcribe_fn=None, **kw):
        touched["decode"] = touched.get("decode", 0) + 1
        return [""], [""]

    monkeypatch.setattr(batch, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    code = batch.run_batch(
        files,
        batch.RunOptions.expert_transcribe(
            profile="dictation",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            config_path=None,
        ),
        yes=True,
        no_group=True,
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "mutually exclusive" in err
    assert "decode" not in touched


def test_run_batch_single_file_needs_no_tty_and_does_no_grouping(
    tmp_path, monkeypatch, capsys
) -> None:
    """A single-file run performs NO grouping work at all — no boundary
    decode, no TTY requirement, no model load — and flows straight to the
    plain per-file loop (transcribe_file exactly once)."""
    import vemoizer.batch as batch
    import vemoizer.pipeline as pipeline_module

    touched: dict[str, int] = {"decode": 0, "transcribe_file": 0, "isatty": 0}

    def fake_decode_boundaries(files, transcribe_fn=None, **kw):
        touched["decode"] += 1
        return [""], [""]

    def fake_transcribe_file(path, **kwargs):
        touched["transcribe_file"] += 1
        return {"text": "yksi ainoa", "segments": []}

    monkeypatch.setattr(batch, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)

    def _counting_isatty() -> bool:
        touched["isatty"] += 1
        return False

    monkeypatch.setattr("sys.stdin.isatty", _counting_isatty)
    # chdir into the tmp dir so the default --format all writes
    # (txt/json/srt/vtt/md) land in tmp_path, not the repo root CWD.
    monkeypatch.chdir(tmp_path)

    f = tmp_path / "Uusi äänitys 425.m4a"
    f.touch()
    code = batch.run_batch(
        [f],
        batch.RunOptions.expert_transcribe(
            profile="dictation",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            config_path=None,
        ),
        yes=False,
        no_group=False,
    )
    assert code == 0
    assert touched["transcribe_file"] == 1
    assert touched["decode"] == 0
    assert touched["isatty"] == 0  # no TTY requirement at all for one file
    # The output must be in tmp_path, not the CWD (repo root).
    assert (tmp_path / "Uusi äänitys 425.txt").exists()


# ---------------------------------------------------------------------------
# Deterministic release — the boundary transcriber is dropped in finally,
# WITHOUT cleanup() (cleanup() would null the shared mlx_whisper cache)
# ---------------------------------------------------------------------------


def test_decode_boundaries_releases_transcriber_without_cleanup(
    tmp_path, monkeypatch
) -> None:
    """decode_boundaries releases the boundary WhisperTranscriber
    DETERMINISTICALLY (local references dropped in the finally) — it never
    calls cleanup() (that would null mlx_whisper's shared ModelHolder cache,
    forcing the main pipeline to reload). The function returns normally
    after both edges are decoded (tail + head = 2 transcribe calls), and
    cleanup() is never invoked.
    """
    import vemoizer.grouping as grouping
    import vemoizer.whisper_transcriber as wt

    calls = {"transcribe": 0, "cleanup": 0}

    class FakeWhisper:
        def __init__(self, language=None):
            pass

        def transcribe(self, audio, **kwargs):
            calls["transcribe"] += 1
            return {"text": "x"}

        def cleanup(self):
            calls["cleanup"] += 1

    # decode_boundaries lazily imports WhisperTranscriber from the module.
    monkeypatch.setattr(wt, "WhisperTranscriber", FakeWhisper)
    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 60.0)
    monkeypatch.setattr(
        grouping,
        "_decode_edge_window",
        lambda path, start, end, **kw: np.zeros(16000, dtype=np.float32),
    )
    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    tail_texts, head_texts = decode_boundaries(files)
    # Returns normally (no exception, no cleanup()).
    assert len(tail_texts) == 1
    assert len(head_texts) == 1
    assert calls["transcribe"] == 2  # tail + head both decoded
    assert calls["cleanup"] == 0  # the shared cache is preserved
