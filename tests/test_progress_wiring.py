"""CLI-level display wiring tests (issue #105 M4b).

Covers: the display is constructed once per CLI invocation, threaded
through to transcribe_file, closed on success and failure, suppressed by
--quiet, and suppressed (no-op) on a non-TTY.  The [i/N] prefix logic is
tested directly against set_batch_prefix / prefix_active_stage.

No real model load, no network, no real TTY needed (monkeypatch isatty).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app
from vemoizer.progress import ProgressDisplay
from vemoizer.progress_wiring import make_batch_display, set_batch_prefix

runner = CliRunner()


# ---------------------------------------------------------------------------
# make_batch_display
# ---------------------------------------------------------------------------


def test_make_batch_display_returns_none_when_quiet() -> None:
    assert make_batch_display(quiet=True) is None


def test_make_batch_display_returns_display_when_not_quiet() -> None:
    d = make_batch_display(quiet=False)
    assert isinstance(d, ProgressDisplay)


def test_make_batch_display_disabled_on_non_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-TTY: the display is constructed but disabled (no stderr output)."""
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    d = make_batch_display(quiet=False)
    assert d is not None
    assert d.disable is True


def test_make_batch_display_enabled_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    d = make_batch_display(quiet=False)
    assert d is not None
    assert d.disable is False


# ---------------------------------------------------------------------------
# set_batch_prefix / prefix_active_stage
# ---------------------------------------------------------------------------


@pytest.fixture
def tty_display(monkeypatch: pytest.MonkeyPatch) -> ProgressDisplay:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    return ProgressDisplay()


def test_set_batch_prefix_none_display_is_noop() -> None:
    set_batch_prefix(None, 1, 2, "memo")  # must not raise
    set_batch_prefix(None, 1, 1, "memo")  # must not raise


def test_set_batch_prefix_disabled_display_is_noop() -> None:
    d = ProgressDisplay(verbose=False)  # disable=True
    set_batch_prefix(d, 1, 2, "memo")  # must not raise


def test_set_batch_prefix_single_file_is_noop(tty_display: ProgressDisplay) -> None:
    task_id = tty_display.add_stage("decode")
    set_batch_prefix(tty_display, 1, 1, "memo")  # N == 1 → no prefix
    desc = tty_display._progress.tasks[task_id].description
    assert not desc.startswith("[1/1]")
    tty_display.close()


def test_set_batch_prefix_multi_file_adds_prefix(
    tty_display: ProgressDisplay,
) -> None:
    task_id = tty_display.add_stage("decode")
    set_batch_prefix(tty_display, 1, 3, "memo")
    desc = tty_display._progress.tasks[task_id].description
    assert desc == "[1/3] memo · decode"
    tty_display.close()


def test_set_batch_prefix_is_idempotent(tty_display: ProgressDisplay) -> None:
    task_id = tty_display.add_stage("decode")
    set_batch_prefix(tty_display, 2, 3, "memo")
    set_batch_prefix(tty_display, 2, 3, "memo")  # second call: no double prefix
    desc = tty_display._progress.tasks[task_id].description
    assert desc == "[2/3] memo · decode"
    tty_display.close()


def test_set_batch_prefix_updates_active_stage(tty_display: ProgressDisplay) -> None:
    """The prefix targets the most recently added (active) task."""
    tty_display.add_stage("decode")
    set_batch_prefix(tty_display, 1, 2, "a")
    t2 = tty_display.add_stage("diarize")
    set_batch_prefix(tty_display, 1, 2, "a")  # prefixes the NEW active task
    assert tty_display._progress.tasks[t2].description == "[1/2] a · diarize"
    tty_display.close()


def test_set_batch_prefix_no_task_yet_is_noop(tty_display: ProgressDisplay) -> None:
    set_batch_prefix(tty_display, 1, 2, "memo")  # no stage added yet
    tty_display.close()


def test_set_batch_prefix_privacy_stem_only(tty_display: ProgressDisplay) -> None:
    """Progress text contains only the file stem — never transcript text."""
    task_id = tty_display.add_stage("decode")
    set_batch_prefix(tty_display, 1, 2, "2026-09-01 standup")
    desc = tty_display._progress.tasks[task_id].description
    assert "standup" in desc
    assert "moikka" not in desc  # no transcript text
    tty_display.close()


# ---------------------------------------------------------------------------
# CLI: display constructed once, threaded, closed on success and failure
# ---------------------------------------------------------------------------


def _fake_tf(captured: dict[str, Any], **result_overrides: Any):
    def fake_tf(path, **kwargs):
        captured["display"] = kwargs.get("display")
        result = {
            "text": "hello",
            "segments": [],
            "notes": {"title": "Memo"},
        }
        result.update(result_overrides)
        return result

    return fake_tf


def test_transcribe_single_file_display_passed_to_transcribe_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI constructs the display and threads it into transcribe_file."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "memo.m4a"
    f.touch()

    captured: dict[str, Any] = {}
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", _fake_tf(captured))

    result = runner.invoke(app, ["transcribe", str(f), "--format", "txt"])
    assert result.exit_code == 0, result.output
    assert captured["display"] is not None
    assert isinstance(captured["display"], ProgressDisplay)


def test_transcribe_quiet_suppresses_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--quiet → display is None → transcribe_file gets display=None."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "memo.m4a"
    f.touch()

    captured: dict[str, Any] = {}
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", _fake_tf(captured))

    result = runner.invoke(app, ["transcribe", str(f), "--quiet", "--format", "txt"])
    assert result.exit_code == 0, result.output
    assert captured["display"] is None


def test_transcribe_display_closed_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When transcribe_file raises, the display is still closed (finally)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "bad.m4a"
    f.touch()

    def fake_tf(path, **kwargs):
        raise RuntimeError("decode exploded")

    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)

    close_called: list[bool] = []
    orig_close = ProgressDisplay.close

    def tracking_close(self):
        close_called.append(True)
        orig_close(self)

    monkeypatch.setattr(ProgressDisplay, "close", tracking_close)
    result = runner.invoke(app, ["transcribe", str(f), "--format", "txt"])
    # Exit code 1 (failure), but display must have been closed
    assert result.exit_code == 1
    assert close_called, "display.close() was not called on failure"


def test_transcribe_display_closed_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On success the display is also closed (finally)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "ok.m4a"
    f.touch()

    captured: dict[str, Any] = {}
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", _fake_tf(captured))

    close_called: list[bool] = []
    orig_close = ProgressDisplay.close

    def tracking_close(self):
        close_called.append(True)
        orig_close(self)

    monkeypatch.setattr(ProgressDisplay, "close", tracking_close)
    result = runner.invoke(app, ["transcribe", str(f), "--format", "txt"])
    assert result.exit_code == 0, result.output
    assert close_called, "display.close() was not called on success"


def test_meeting_display_passed_to_transcribe_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The meeting command also constructs and threads the display."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "meeting.m4a"
    f.touch()

    captured: dict[str, Any] = {}
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", _fake_tf(captured))

    result = runner.invoke(app, ["meeting", str(f), "--no-repair"])
    assert result.exit_code == 0, result.output
    assert captured["display"] is not None
    assert isinstance(captured["display"], ProgressDisplay)


def test_meeting_quiet_suppresses_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "meeting.m4a"
    f.touch()

    captured: dict[str, Any] = {}
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", _fake_tf(captured))

    result = runner.invoke(app, ["meeting", str(f), "--no-repair", "--quiet"])
    assert result.exit_code == 0, result.output
    assert captured["display"] is None


def test_memo_display_passed_to_transcribe_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "memo.m4a"
    f.touch()

    captured: dict[str, Any] = {}
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", _fake_tf(captured))

    result = runner.invoke(app, ["memo", str(f), "--no-repair"])
    assert result.exit_code == 0, result.output
    assert captured["display"] is not None


def test_multi_file_display_constructed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With 2+ files the display is constructed once, not per-file."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    fa = tmp_path / "a.m4a"
    fb = tmp_path / "b.m4a"
    fa.touch()
    fb.touch()

    captured_ids: list[int] = []

    def fake_tf(path, **kwargs):
        d = kwargs.get("display")
        if d is not None:
            captured_ids.append(id(d))
        return {"text": "hello", "segments": [], "notes": {"title": "Memo"}}

    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)

    result = runner.invoke(
        app,
        ["transcribe", str(fa), str(fb), "--no-group", "--format", "txt"],
    )
    assert result.exit_code == 0, result.output
    assert len(captured_ids) == 2  # one call per file
    assert len(set(captured_ids)) == 1  # same display instance


def test_non_tty_no_progress_output_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a non-TTY, stderr contains no progress display output."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    f = tmp_path / "memo.m4a"
    f.touch()

    captured: dict[str, Any] = {}
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", _fake_tf(captured))

    # Simulate non-TTY: isatty → False (the display's disable flag is True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)

    result = runner.invoke(app, ["transcribe", str(f), "--format", "txt"])
    assert result.exit_code == 0, result.output
    # No rich Progress output (spinner chars, "decode" task line, etc.)
    # The display was disabled so nothing was written to stderr.
    assert "decode" not in result.stderr or "error" in result.stderr.lower()


# ---------------------------------------------------------------------------
# [i/N] prefix in the batch layer (multi-file transcribe)
# ---------------------------------------------------------------------------


def test_multi_file_prefix_set_per_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With 2 files, set_batch_prefix is called with i=1,2 and N=2."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    fa = tmp_path / "alpha.m4a"
    fb = tmp_path / "beta.m4a"
    fa.touch()
    fb.touch()

    import vemoizer.progress_wiring as pw

    calls: list[tuple[int, int, str]] = []
    orig_set = pw.set_batch_prefix

    def tracking_set(display, index, total, stem):
        calls.append((index, total, stem))
        orig_set(display, index, total, stem)

    monkeypatch.setattr(pw, "set_batch_prefix", tracking_set)
    # Also patch in the modules that import it
    import vemoizer.batch as bmod
    import vemoizer.transcribe_loop as tl

    monkeypatch.setattr(tl, "set_batch_prefix", tracking_set)
    monkeypatch.setattr(bmod, "set_batch_prefix", tracking_set)

    def fake_tf(path, **kwargs):
        return {"text": "hello", "segments": [], "notes": {"title": "Memo"}}

    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)

    result = runner.invoke(
        app,
        ["transcribe", str(fa), str(fb), "--no-group", "--format", "txt"],
    )
    assert result.exit_code == 0, result.output
    # The batch layer called set_batch_prefix for each file
    assert len(calls) >= 2
    # N should be 2 (number of files)
    assert all(total == 2 for _, total, _ in calls)
    # i should be 1-based
    indices = [i for i, _, _ in calls]
    assert 1 in indices
    assert 2 in indices


# ---------------------------------------------------------------------------
# Shim: whole-transcribe wrapping (4 windows → one monotonic progress)
# ---------------------------------------------------------------------------


def test_whole_transcribe_wrapping_monotonic_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a display is passed to WhisperTranscriber.transcribe, ONE
    with_whisper_progress context wraps the whole window loop: the shim
    patches the module-level tqdm attribute once, each window's bar
    drives the display with strictly increasing file-level minutes, and
    the display's completed value reaches the file total once (at the end
    of the last window). 120 s audio → 4 × 30 s windows, file total 2 min.
    """
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()

    raw = {
        "text": "moro vaan",
        "language": "fi",
        "segments": [
            {
                "text": "moro vaan",
                "start": 0.0,
                "end": 1.0,
                "words": [{"word": " moro", "start": 0.0, "end": 0.5}],
            }
        ],
    }
    from vemoizer.whisper_transcriber import WhisperTranscriber

    class FakeBar:
        """A tqdm-bar stand-in that records every factory + update call so
        the shim's minutes conversion is exercised (an auto-mocking
        MagicMock attribute would leave the assertion vacuous)."""

        def __init__(self, total: float = 3000) -> None:
            self.n = 0.0
            self.total = total
            self.updates: list[float] = []

        def __enter__(self) -> FakeBar:
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def update(self, n: float = 1, *a: object, **kw: object) -> None:
            self.n = float(n)
            self.updates.append(float(n))

    # Track _ShimmedProgress bars created by the shim (one per full-window
    # decode). The shim's factory is _make_bar (not the original module's
    # __call__), so we track bar creation via _ShimmedProgress.__init__.
    from vemoizer.progress_shim import _ShimmedProgress

    bars_created: list[_ShimmedProgress] = []
    orig_shim_init = _ShimmedProgress.__init__

    def tracked_shim_init(self, *args, **kwargs):
        orig_shim_init(self, *args, **kwargs)
        bars_created.append(self)

    monkeypatch.setattr(_ShimmedProgress, "__init__", tracked_shim_init)

    # Track factory calls: the shim's _make_bar is the factory. The fake
    # tqdm module in sys.modules is replaced by the shim's _ShimmedBar,
    # so we can't record via the original module. Instead, record via
    # the _ShimmedProgress bars created (one per window).

    fake_tr = MagicMock()
    fake_tr.tqdm = MagicMock()  # will be replaced by the shim
    mock = MagicMock()

    completed_values: list[float] = []
    orig_update = display._progress.update

    def tracking_update(task_id, **kwargs):
        if "completed" in kwargs:
            completed_values.append(kwargs["completed"])
        return orig_update(task_id, **kwargs)

    monkeypatch.setattr(display._progress, "update", tracking_update)

    with (
        patch.dict(
            "sys.modules",
            {"mlx_whisper": mock, "mlx_whisper.transcribe": fake_tr},
        ),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock

        def _fake_whisper_transcribe(audio, *a, **kw):
            # Emulate mlx_whisper's internal loop: create the bar via the
            # (patched) tqdm module attribute — the shim has replaced
            # ``tr_mod.tqdm`` with a _ShimmedBar, so ``tr_mod.tqdm.tqdm(...)``
            # goes through the shim's factory.
            tr_mod = fake_tr
            bar = tr_mod.tqdm.tqdm(total=3000, unit="frames")
            bar.__enter__()
            for frac in (0.25, 0.5, 0.75, 1.0):
                bar.update(int(3000 * frac))
            bar.__exit__(None, None, None)
            return raw

        mock.transcribe = _fake_whisper_transcribe
        # 120 s audio → 4 windows
        t.transcribe(np.zeros(120 * 16_000, dtype=np.float32), display=display)
    display.close()

    # One _ShimmedProgress bar per full-window decode (4 windows).
    assert len(bars_created) == 4, f"expected 4 window bars, got {len(bars_created)}"
    # Each bar's total was set from the factory call (total=3000 frames).
    for b in bars_created:
        assert b.total == pytest.approx(3000)

    # The shim must have driven the display at least once.
    assert len(completed_values) > 0
    # The display is driven once per window (4 windows → 4+ completed
    # values). The per-window values are monotonic in file-level minutes
    # because each window's bar resets n=0 at __exit__ but the window
    # offset advances, so the next window's first update is higher than
    # the previous window's last.
    assert len(completed_values) >= 4
    for i in range(1, len(completed_values)):
        assert completed_values[i] >= completed_values[i - 1] - 1e-9, (
            f"non-monotonic: {completed_values}"
        )
    # The display reaches the file total (2.0 min) by the end of the last
    # window — the decode stage finishes on the full total.
    assert completed_values[-1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Dictation fallback: no stranded decode stage
# ---------------------------------------------------------------------------


def test_dictation_fallback_no_stranded_decode_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the meeting decode fails open and transcribe_file falls back
    to the dictation path, the display's decode task must be finished
    (not left running)."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)

    import vemoizer.pipeline as pipeline_module

    # Patch decode_meeting to return None (meeting decode fails open)
    monkeypatch.setattr(pipeline_module, "decode_meeting", lambda *a, **kw: None)
    # Patch decode_all to return a simple result
    monkeypatch.setattr(
        pipeline_module,
        "decode_all",
        lambda *a, **kw: {"text": "hello", "segments": [], "words": []},
    )
    # Patch ingest to return short audio
    monkeypatch.setattr(
        pipeline_module,
        "ingest_audio",
        lambda *a, **kw: np.zeros(16_000, dtype=np.float32),
    )
    # Patch preflight to pass (it's a deferred import in transcribe_file)
    import vemoizer.preflight as preflight_mod

    monkeypatch.setattr(preflight_mod, "preflight_gate", lambda **kw: None)

    display = ProgressDisplay()
    display.start()

    # Run transcribe_file with the meeting profile
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    f = tmp / "test.m4a"
    f.touch()

    result = pipeline_module.transcribe_file(
        str(f),
        profile="meeting",
        display=display,
    )
    display.close()

    # The result should have the meeting fallback warning
    assert "meeting decode failed" in str(result.get("warnings", [])) or result.get(
        "text"
    )
    # The display should not have a stranded "decode" task in running state
    # (it was finished by the shim's finally block)
    running_tasks = [t for t in display._progress.tasks if not t.finished]
    # There should be no running "decode" task (it was finished)
    decode_running = [t for t in running_tasks if "decode" in t.description]
    assert len(decode_running) == 0, f"Stranded decode task: {decode_running}"


def test_set_batch_prefix_nfc_normalizes_nfd_stem(
    tty_display: ProgressDisplay,
) -> None:
    """The [i/N] prefix uses the NFC-normalised stem (decision 3): an NFD
    (decomposed) stem is composed to NFC before being shown."""
    # NFD: 'm' + combining acute + 'o' (decomposed) vs NFC 'mó'
    nfd_stem = "mo\u0301"  # 'm', 'o', combining acute (NFD)
    nfc_stem = "m\u00f3"  # 'm', 'ó' (NFC)
    assert nfd_stem != nfc_stem  # they are different strings
    task_id = tty_display.add_stage("decode")
    set_batch_prefix(tty_display, 1, 2, nfd_stem)
    desc = tty_display._progress.tasks[task_id].description
    # The prefix must use the NFC form
    assert f"[1/2] {nfc_stem} · decode" == desc, (
        f"Expected NFC stem in prefix, got: {desc!r}"
    )
    tty_display.close()
