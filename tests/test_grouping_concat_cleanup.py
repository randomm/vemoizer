"""Regression tests for ``grouping_concat`` (issue #77 round-4 corrections).

Covered here (moved out of ``test_grouping_bugs.py`` for the 800-line test
file limit):

- ``concat_groups`` failure paths: ffmpeg non-zero exit, ``TimeoutExpired``,
  an unexpected exception, and ``KeyboardInterrupt`` each remove the
  PARTIAL merged file (``-y`` may have left bytes behind) and the 0o700
  temp dir; the success path leaves the merged file for the caller.
- ``part_offsets`` budget semantics: the cumulative wall-clock budget
  shrinks only by real elapsed ``time.monotonic()`` time (driven by a fake
  clock), never by the parts' audio durations — two 30-minute parts that
  decode in milliseconds must not exhaust the 900 s budget. Exhaustion
  raises ``IngestError`` naming the part and the budget.
- ``concat_groups`` newline / carriage-return filename rejection.

Helpers are shared from ``test_grouping_bugs`` (import, no copy-paste).
No model loads, no network; the temp dir is pinned into ``tmp_path`` by
monkeypatching ``tempfile.mkdtemp`` — production code carries no test
seam.
"""

from __future__ import annotations

import json
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest
from test_grouping_bugs import _make_wav

import vemoizer.grouping as grouping
import vemoizer.grouping_concat as gc
from vemoizer.grouping import GroupingError
from vemoizer.ingest import IngestError
from vemoizer.presets import RunOptions


def _pin_mkdtemp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pin the concat temp dir into *tmp_path* by faking ``tempfile.mkdtemp``.

    The fake creates and returns the pinned dir, so the test can assert on
    it without globbing the system temp dir.
    """
    d = tmp_path / "vemoizer-concat-pin"
    d.mkdir()

    def fake_mkdtemp(**kwargs: str) -> str:
        return str(d)

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)
    return d


def _probe_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every part look like the same audio stream (a passthrough for
    the per-part ffprobe signature check)."""
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "aac,48000,1")


def _two_wavs(tmp_path: Path) -> list[Path]:
    return [
        _make_wav(tmp_path / "Uusi äänitys 425.wav", 0.5),
        _make_wav(tmp_path / "Uusi äänitys 426.wav", 0.5),
    ]


# ---------------------------------------------------------------------------
# concat_groups failure paths: the partial merged file AND the temp dir
# must be gone (ffmpeg -y may have left partial bytes before the error).
# ---------------------------------------------------------------------------


def test_concat_groups_nonzero_exit_removes_partial_file_and_dir(
    tmp_path, monkeypatch
) -> None:
    """ffmpeg -y writes a partial group file and then exits non-zero:
    the partial file AND the 0o700 temp dir must be gone, and a
    GroupingError is raised."""
    tmp = _pin_mkdtemp(monkeypatch, tmp_path)
    _probe_ok(monkeypatch)
    a, b = _two_wavs(tmp_path)

    class FakeProc:
        returncode = 1
        stdout = b""
        stderr = b"concat: some failure"

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"\x00partial private audio")
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GroupingError):
        gc.concat_groups([a, b])
    assert not (tmp / "group.wav").exists()
    assert not tmp.exists()


def test_concat_groups_timeout_removes_partial_file_and_dir(
    tmp_path, monkeypatch
) -> None:
    """ffmpeg -y writes a partial group file and then the timeout expires:
    the partial file AND the temp dir must be gone, and a GroupingError is
    raised."""
    tmp = _pin_mkdtemp(monkeypatch, tmp_path)
    _probe_ok(monkeypatch)
    a, b = _two_wavs(tmp_path)

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"\x00partial private audio")
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120.0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GroupingError):
        gc.concat_groups([a, b])
    assert not (tmp / "group.wav").exists()
    assert not tmp.exists()


def test_concat_groups_unexpected_error_removes_partial_file_and_dir(
    tmp_path, monkeypatch
) -> None:
    """An unexpected exception (non-GroupingError) after the partial file
    exists: the original exception propagates unchanged AND both the
    partial file and the temp dir are gone."""
    tmp = _pin_mkdtemp(monkeypatch, tmp_path)
    _probe_ok(monkeypatch)
    a, b = _two_wavs(tmp_path)

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"\x00partial private audio")
        raise RuntimeError("boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="boom"):
        gc.concat_groups([a, b])
    assert not (tmp / "group.wav").exists()
    assert not tmp.exists()


def test_concat_groups_keyboard_interrupt_removes_partial_file_and_dir(
    tmp_path, monkeypatch
) -> None:
    """A KeyboardInterrupt after the partial file exists: the signal
    propagates unchanged AND both the partial file and the temp dir are
    gone."""
    tmp = _pin_mkdtemp(monkeypatch, tmp_path)
    _probe_ok(monkeypatch)
    a, b = _two_wavs(tmp_path)

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"\x00partial private audio")
        raise KeyboardInterrupt()

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(KeyboardInterrupt):
        gc.concat_groups([a, b])
    assert not (tmp / "group.wav").exists()
    assert not tmp.exists()


def test_concat_groups_success_leaves_file_for_caller(tmp_path, monkeypatch) -> None:
    """On success the merged file and its temp dir must survive the call:
    the caller (run_batch) owns them and removes them via
    remove_concat_output afterwards."""
    tmp = _pin_mkdtemp(monkeypatch, tmp_path)
    _probe_ok(monkeypatch)
    a, b = _two_wavs(tmp_path)

    class FakeProc:
        returncode = 0
        stdout = b""
        stderr = b""

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"merged audio")
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = gc.concat_groups([a, b])
    assert out == tmp / "group.wav"
    assert out.is_file()
    assert tmp.is_dir()
    # The list file is gone even on success.
    assert not (tmp / "concat.txt").exists()
    # The caller's cleanup still removes the file AND the dir.
    gc.remove_concat_output(out)
    assert not tmp.exists()


def test_concat_groups_success_temp_dir_is_mode_700_and_list_gone(
    tmp_path, monkeypatch
) -> None:
    """The concat temp dir is provably 0o700 (``os.chmod`` right after
    mkdtemp, defence in depth) and the concat list file does not outlive
    the call — the merged file itself survives for the caller."""
    tmp = _pin_mkdtemp(monkeypatch, tmp_path)
    _probe_ok(monkeypatch)
    a, b = _two_wavs(tmp_path)

    class FakeProc:
        returncode = 0
        stdout = b""
        stderr = b""

    def fake_run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"merged audio")
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = gc.concat_groups([a, b])

    # The merged file survives the call (the caller owns it)...
    assert out == tmp / "group.wav"
    assert out.is_file()
    # ...and the temp dir holding it is 0o700, mode bits exactly.
    assert stat.S_IMODE(tmp.stat().st_mode) == 0o700
    # ...while the concat list file does not outlive the call.
    assert not (tmp / "concat.txt").exists()

    gc.remove_concat_output(out)
    assert not tmp.exists()


# ---------------------------------------------------------------------------
# part_offsets budget semantics: the cumulative wall-clock budget shrinks
# only by real elapsed time.monotonic() time, never by audio durations.
# ---------------------------------------------------------------------------


class _FakeClock:
    """A controllable ``time.monotonic`` stand-in for the budget loop."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _fake_pcm_for_clock(
    clock: _FakeClock,
    durations: list[float],
    timeouts_seen: list[float],
    stall_all: bool = False,
    wall_costs: list[float] | None = None,
):
    """A fake ``pcm_duration_seconds``: reports the pre-set audio *durations*
    and advances the shared fake clock by the per-decode wall cost.
    With *stall_all*, each decode takes its full granted timeout. Otherwise
    the wall cost is taken from *wall_costs* (or *durations* as a default)."""
    idx = {"i": 0}
    costs = wall_costs if wall_costs is not None else durations

    def fake_pcm(path, timeout=300.0, **kw):
        i = idx["i"]
        idx["i"] += 1
        timeouts_seen.append(timeout)
        cost = timeout if stall_all else costs[i]
        clock.advance(cost)
        return durations[i]

    return fake_pcm


def test_part_offsets_budget_shrinks_only_by_elapsed_wall_clock(
    tmp_path, monkeypatch
) -> None:
    """With a fake clock, each decode 'takes' its full granted timeout of
    wall time, so the ``timeout=`` value passed to the next decode shrinks
    900 -> 500 -> 100, and a 4th part finds the budget exhausted: an
    IngestError naming the part and the budget."""
    files = [tmp_path / f"Uusi äänitys {425 + i}.m4a" for i in range(4)]
    for f in files:
        f.touch()
    clock = _FakeClock()
    seen: list[float] = []
    # Each decode 'takes' its full granted timeout of wall time (a decode
    # that runs to its timeout has, by definition, spent that wall time),
    # while reporting only 0.1 s of AUDIO — the budget must still shrink
    # 900 -> 500 -> 100 by wall time, not by audio duration.
    monkeypatch.setattr(
        grouping,
        "pcm_duration_seconds",
        _fake_pcm_for_clock(
            clock,
            [0.1, 0.1, 0.1, 0.1],
            seen,
            stall_all=False,
            wall_costs=[400.0, 400.0, 400.0],
        ),
    )
    monkeypatch.setattr(gc.time, "monotonic", clock)

    with pytest.raises(IngestError) as exc:
        gc.part_offsets(files, total_timeout=900.0)

    # Timeouts shrink by the fake wall-clock cost, not by audio duration.
    assert seen == pytest.approx([900.0, 500.0, 100.0])
    msg = str(exc.value)
    assert "Uusi äänitys 428.m4a" in msg  # names the 4th, unmeasured part
    assert "900" in msg  # names the total budget


def test_part_offsets_stalled_part_exhausts_budget(tmp_path, monkeypatch) -> None:
    """A stalled decode that consumes the ENTIRE remaining budget leaves
    nothing for the next part: IngestError naming the next part and the
    budget."""
    files = [
        tmp_path / "Uusi äänitys 425.m4a",
        tmp_path / "Uusi äänitys 426.m4a",
    ]
    for f in files:
        f.touch()
    clock = _FakeClock()
    seen: list[float] = []
    monkeypatch.setattr(
        grouping,
        "pcm_duration_seconds",
        _fake_pcm_for_clock(clock, [1.0, 1.0], seen, stall_all=True),
    )
    monkeypatch.setattr(gc.time, "monotonic", clock)

    with pytest.raises(IngestError) as exc:
        gc.part_offsets(files, total_timeout=60.0)

    assert seen == pytest.approx([60.0])  # only part 1 got to decode
    msg = str(exc.value)
    assert "Uusi äänitys 426.m4a" in msg
    assert "60" in msg


def test_part_offsets_two_long_parts_do_not_exhaust_budget(
    tmp_path, monkeypatch
) -> None:
    """The bug: two 30-minutes-of-AUDIO parts decoded in milliseconds each
    must NOT exhaust the 900 s budget (the deadline shrinks only by real
    elapsed wall time, never by the parts' audio durations). The old code
    subtracted each part's duration from the deadline and raised on part 2."""
    files = [
        tmp_path / "Uusi äänitys 425.m4a",
        tmp_path / "Uusi äänitys 426.m4a",
        tmp_path / "Uusi äänitys 427.m4a",
    ]
    for f in files:
        f.touch()
    # Each part is 1800 s of audio, but the fake clock advances by only a
    # fraction of a second of wall time per decode.
    clock = _FakeClock()
    seen: list[float] = []
    monkeypatch.setattr(
        grouping,
        "pcm_duration_seconds",
        _fake_pcm_for_clock(
            clock,
            [1800.0, 1800.0, 1800.0],
            seen,
            stall_all=False,
            wall_costs=[0.001, 0.001, 0.001],
        ),
    )
    monkeypatch.setattr(gc.time, "monotonic", clock)

    offsets = gc.part_offsets(files, total_timeout=900.0)
    assert [o.start_offset for o in offsets] == pytest.approx([0.0, 1800.0, 3600.0])
    # No error, and the first decode got (nearly) the full budget.
    assert seen[0] == pytest.approx(900.0, abs=1.0)


# ---------------------------------------------------------------------------
# Newline / carriage-return in a part file name: rejected up front, before
# the concat list file is written.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name", ["Uusi äänitys 425\n.m4a", "Uusi äänitys 425\r.m4a"]
)
def test_concat_groups_rejects_newline_in_filename(
    tmp_path, monkeypatch, bad_name
) -> None:
    """A part whose name contains '\\n' or '\\r' is rejected up front with
    a GroupingError naming the file — the concat list file is never
    written and no temp dir is created."""
    tmp = _pin_mkdtemp(monkeypatch, tmp_path)
    _probe_ok(monkeypatch)
    a = _make_wav(tmp_path / "Uusi äänitys 425.wav", 0.5)
    b = tmp_path / bad_name
    b.touch()

    def fake_run(argv, **kwargs):
        raise AssertionError("ffmpeg must not run for a newline filename")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(GroupingError) as exc:
        gc.concat_groups([a, b])
    assert "Uusi äänitys 425" in str(exc.value)
    assert "newline" in str(exc.value)
    # Rejected before any list file or merged file was written.
    assert not (tmp / "concat.txt").exists()


# ---------------------------------------------------------------------------
# M5a: the preset write seam's duration measurement is fail-open
# ---------------------------------------------------------------------------


def test_run_preset_group_duration_ingest_error_is_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group whose parts raise IngestError in ``group_durations``: the
    seam skips the durations (no ``duration_s`` in the sidecar) and the
    run still succeeds with a full .md + .json pair (issue #89)."""
    import vemoizer.batch_preset as batch_preset_module
    import vemoizer.grouping as grouping
    import vemoizer.sidecar as sidecar_module

    def boom(paths, **kw):
        raise IngestError("ffmpeg failed to decode group parts")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sidecar_module, "group_durations", boom)
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None, **kw: ([""], ["x"]),
    )
    monkeypatch.setattr(
        grouping, "propose_groups", lambda files, t, h: [[files[0]], [files[1]]]
    )
    monkeypatch.setattr(
        grouping, "confirm_groups", lambda files, proposals, **kwargs: list(proposals)
    )
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kwargs: {"text": "moikka", "segments": []},
    )

    a = _make_wav(tmp_path / "a.wav", 0.5)
    b = _make_wav(tmp_path / "b.wav", 0.5)

    options = RunOptions.expert_transcribe(
        profile="dictation",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )
    code = batch_preset_module._run_preset_groups(
        [a, b],
        options,
        quiet=True,
        yes=True,
        no_group=False,
        transcribe_fn=lambda path, **kwargs: {"text": "moikka", "segments": []},
        input_fn=None,
        print_fn=None,
        tty_isatty=lambda: True,
        effective_glossary=None,
        command="meeting",
    )
    assert code == 0
    # One pair per group (two single-part groups), written despite the
    # duration failure.
    md_files = list(tmp_path.glob("*.md"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(md_files) == 2
    assert len(json_files) == 2
    # No duration_s key in any sidecar (fail-open: skip on IngestError).
    for json_file in json_files:
        sidecar_data = json.loads(json_file.read_text(encoding="utf-8"))
        for entry in sidecar_data.get("source", []):
            assert "duration_s" not in entry


def _fake_preset_groups(monkeypatch: pytest.MonkeyPatch, groups) -> None:
    """Pin the M3 grouping flow to fixed *groups* (no boundary decodes)."""
    import vemoizer.grouping as grouping

    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None, **kw: ([""], ["x"]),
    )
    monkeypatch.setattr(grouping, "propose_groups", lambda files, t, h: groups)
    monkeypatch.setattr(
        grouping, "confirm_groups", lambda files, proposals, **kwargs: list(proposals)
    )
    # Every part looks like the same audio stream (the concat probe), so the
    # multi-part concat proceeds past the stream-signature check.
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "wav,16000,1")
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kwargs: {"text": "moikka", "segments": []},
    )


def test_run_preset_group_durations_use_real_part_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grouped run, CWD different from the files' directory: the seam's
    ``group_durations`` must measure the FULL part paths (resolved via
    ``group_part_paths``), so ``source[].duration_s`` is present for
    every part (issue #89 regression: bare names resolved against the CWD
    failed and ``duration_s`` was silently dropped)."""
    import vemoizer.batch_preset as batch_preset_module

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    a = _make_wav(tmp_path / "a.wav", 0.5)
    b = _make_wav(tmp_path / "b.wav", 0.5)
    c = _make_wav(tmp_path / "c.wav", 0.5)
    monkeypatch.chdir(cwd)

    durations = {
        str(a): 11.0,
        str(b): 22.0,
        str(c): 33.0,
    }

    def fake_pcm(path, **kwargs):
        full = str(path)
        if full in durations:
            return durations[full]
        raise IngestError(f"cannot decode {path!r} (not a full path)")

    import vemoizer.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "pcm_duration_seconds", fake_pcm)
    # Three single-part groups. Each part lives in tmp_path (CWD is the
    # separate tmp_path/cwd dir), so a bare-name/CWD-relative resolution
    # would hit a path fake_pcm rejects; only the real full paths succeed.
    # (Multi-file single-part groups use str labels, resolved via
    # group_part_paths against the run's files list.)
    _fake_preset_groups(monkeypatch, [[a], [b], [c]])

    options = RunOptions.expert_transcribe(
        profile="dictation",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )
    code = batch_preset_module._run_preset_groups(
        [a, b, c],
        options,
        quiet=True,
        yes=True,
        no_group=False,
        transcribe_fn=lambda path, **kwargs: {"text": "moikka", "segments": []},
        input_fn=None,
        print_fn=None,
        tty_isatty=lambda: True,
        effective_glossary=None,
        command="meeting",
    )
    assert code == 0
    # One sidecar pair per group (three single-part groups), each carrying
    # the part's real path and its measured duration (keyed by full path).
    json_files = {
        json_file.read_text(encoding="utf-8") for json_file in cwd.glob("*.json")
    }
    by_path = {}
    for raw in json_files:
        data = json.loads(raw)
        for entry in data.get("source", []):
            by_path[entry["path"]] = entry["duration_s"]
    # duration_s present for EVERY part, keyed by the FULL part path.
    assert by_path == {str(a): 11.0, str(b): 22.0, str(c): 33.0}


def test_run_preset_group_resolves_parts_against_files_not_same_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-part group's str label (``a.wav+b.wav``) resolves each part
    against the run's ``files`` list — not against the CWD — so when a
    same-basename file exists in *another* directory, ``source[].path``
    and ``duration_s`` point at the correct on-disk part (issue #89
    regression: bare names resolved against the CWD failed and
    ``duration_s`` was silently dropped)."""
    import vemoizer.batch_preset as batch_preset_module

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (tmp_path / "sub").mkdir()
    a = _make_wav(tmp_path / "sub" / "a.wav", 0.5)
    b = _make_wav(tmp_path / "sub" / "b.wav", 0.5)
    # A same-basename file in ANOTHER directory. The group's parts are
    # sub/a.wav + sub/b.wav; if a bare name ever resolved against the CWD
    # or against a different directory, the measurement would hit this
    # decoy (or fail). The group is multi-part, so its label is the str
    # "a.wav+b.wav" and resolution goes through group_part_paths.
    (tmp_path / "decoy").mkdir()
    decoy_a = _make_wav(tmp_path / "decoy" / "a.wav", 0.5)
    decoy_b = _make_wav(tmp_path / "decoy" / "b.wav", 0.5)
    monkeypatch.chdir(cwd)

    durations = {str(a): 11.0, str(b): 22.0}
    measured: list[str] = []

    def fake_pcm(path, **kwargs):
        full = str(path)
        measured.append(full)
        if full in durations:
            return durations[full]
        # Any CWD-relative or decoy path is the bug — reject it so the
        # test fails loudly rather than silently dropping duration_s.
        raise IngestError(f"unexpected path {path!r}")

    import vemoizer.grouping as grouping
    import vemoizer.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "pcm_duration_seconds", fake_pcm)
    # part_offsets measures the ORIGINAL parts (not the merged file) via
    # the grouping re-export — patch that alias to the same fake so the
    # tiny fixture parts don't go to real ffmpeg.
    monkeypatch.setattr(grouping, "pcm_duration_seconds", fake_pcm)
    # Fake the ffmpeg concat (the seam under test is duration/path
    # resolution, not the demuxer itself): the merged file is the group's
    # first part (single-file passthrough semantics the seam relies on).
    import subprocess

    class _FakeProc:
        returncode = 0
        stdout = b""
        stderr = b""

    def _fake_run(argv, **kwargs):
        # The merged file just needs to exist (the transcribe is faked and
        # part_offsets uses the patched fake, not a real decode of it).
        Path(argv[-1]).write_bytes(b"merged audio")
        return _FakeProc()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    _fake_preset_groups(monkeypatch, [[a, b]])

    options = RunOptions.expert_transcribe(
        profile="dictation",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )
    code = batch_preset_module._run_preset_groups(
        [a, b, decoy_a, decoy_b],
        options,
        quiet=True,
        yes=True,
        no_group=False,
        transcribe_fn=lambda path, **kwargs: {"text": "moikka", "segments": []},
        input_fn=None,
        print_fn=None,
        tty_isatty=lambda: True,
        effective_glossary=None,
        command="meeting",
    )
    assert code == 0
    json_files = list(cwd.glob("*.json"))
    assert len(json_files) == 1
    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    entries = data.get("source", [])
    assert len(entries) == 2
    # Both parts resolve to the sub/ files (not the CWD, not the decoys),
    # in marker order, each with its own measured duration.
    assert entries[0]["path"] == str(a)
    assert entries[0]["duration_s"] == 11.0
    assert entries[1]["path"] == str(b)
    assert entries[1]["duration_s"] == 22.0
    # Only the real parts were measured (no CWD-relative / decoy path).
    # fake_pcm is called once by part_offsets and once by group_durations
    # per part, so each path appears twice — assert on the distinct set.
    assert set(measured) == {str(a), str(b)}
