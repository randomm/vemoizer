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

import subprocess
import tempfile
from pathlib import Path

import pytest
from test_grouping_bugs import _make_wav

import vemoizer.grouping as grouping
import vemoizer.grouping_concat as gc
from vemoizer.grouping import GroupingError
from vemoizer.ingest import IngestError


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

    def fake_pcm(path, timeout=300.0):
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
    assert not (tmp / "group.wav").exists()
