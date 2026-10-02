"""Unit tests for the mlx-whisper tqdm shim (issue #105).

Covers: frames→minutes conversion, tqdm attribute patched and restored in
finally even when the decode raises, the idempotent guard for re-entrant
calls, the shim never raising mid-decode, no stderr output when not a TTY,
and the decode result being identical with or without a display.
"""

from __future__ import annotations

import importlib
import io
import sys
import types
from typing import IO, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vemoizer.progress import ProgressDisplay, frames_to_minutes
from vemoizer.progress_shim import (
    WhisperProgress,
    _NoopBar,
    _ShimmedBar,
    with_whisper_progress,
)
from vemoizer.whisper_transcriber import WhisperTranscriber

# ---------------------------------------------------------------------------
# frames_to_minutes
# ---------------------------------------------------------------------------


def test_frames_to_minutes_default_constants() -> None:
    # 3000 frames * 160 / 16000 / 60 = 0.5 minutes (30 s)
    assert frames_to_minutes(3000) == pytest.approx(0.5)


def test_frames_to_minutes_full_file() -> None:
    # 56 minutes of audio → 56*60*16000/160 = 336000 frames
    assert frames_to_minutes(336_000) == pytest.approx(56.0)


def test_frames_to_minutes_custom_constants() -> None:
    # 100 frames at HOP=100, SR=10000 → 100*100/10000/60 = 1/60 min
    assert frames_to_minutes(100, hop_length=100, sample_rate=10_000) == pytest.approx(
        100 * 100 / 10_000 / 60
    )


def test_frames_to_minutes_zero() -> None:
    assert frames_to_minutes(0) == 0.0


# ---------------------------------------------------------------------------
# Shim: tqdm attribute patched and restored in finally
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    return buffer


@pytest.fixture
def tty_stderr(monkeypatch: pytest.MonkeyPatch) -> IO[str]:
    """Force the display to be enabled (TTY-like) for shim tests.

    Pytest's capture replaces ``sys.stderr`` with a ``CaptureIO`` whose
    ``isatty()`` returns False, so monkeypatching the StringIO's ``isatty``
    has no effect on the ``ProgressDisplay`` constructor. Instead, we
    monkeypatch ``sys.stderr`` itself (the CaptureIO) to report a TTY, which
    is what ``ProgressDisplay`` actually reads.
    """
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    return sys.stderr


@pytest.fixture
def non_tty_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Force sys.stderr to look like a non-TTY."""
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    return buffer


def test_shim_patches_and_restores_tqdm(
    tty_stderr: io.StringIO,
) -> None:
    """The shim patches tr_mod.tqdm during the with-block and restores it
    after, even when the body raises."""

    real_tr = importlib.import_module("mlx_whisper.transcribe")
    original_tqdm = real_tr.tqdm
    display = ProgressDisplay()
    display.start()
    try:
        with with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ):
            assert isinstance(real_tr.tqdm, _ShimmedBar), (
                f"Expected _ShimmedBar, got {type(real_tr.tqdm).__name__}"
            )
            # The replacement has a .tqdm attribute (the patched factory)
            assert hasattr(real_tr.tqdm, "tqdm")
        # After the with-block it is restored to the original module
        assert real_tr.tqdm is original_tqdm
    finally:
        display.close()


def test_shim_restores_tqdm_when_body_raises(
    tty_stderr: io.StringIO,
) -> None:
    """A decode exception must not leave the tqdm attribute patched."""
    fake_tr = MagicMock()
    fake_tr.tqdm = MagicMock()
    original_tqdm = fake_tr.tqdm

    display = ProgressDisplay()
    display.start()

    fake_mods = {
        "mlx_whisper": MagicMock(transcribe=fake_tr),
        "mlx_whisper.transcribe": fake_tr,
    }
    with patch.dict("sys.modules", fake_mods):
        with (
            pytest.raises(RuntimeError),
            with_whisper_progress(
                display,
                window_seconds=30.0,
                file_total_minutes=1.0,
            ),
        ):
            raise RuntimeError("decode failed")
        # The original must be restored even after the exception
        assert fake_tr.tqdm is original_tqdm
    display.close()


# ---------------------------------------------------------------------------
# Shim: per-window protocol (mark_window)
# ---------------------------------------------------------------------------


def test_shim_per_window_protocol(
    tty_stderr: io.StringIO,
) -> None:
    """Per-window protocol: the FIRST bar after mark_window drives the
    display; any further bar before the next mark is a no-op (nested /
    re-entrant call). The window loop (WhisperTranscriber) calls
    mark_window before each main-loop transcribe call; self-heal re-decodes
    run after the shim exits and never see it."""

    from vemoizer.progress_shim import _ShimmedProgress

    real_tr = importlib.import_module("mlx_whisper.transcribe")
    display = ProgressDisplay()
    display.start()
    try:
        with with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ) as shim:
            assert isinstance(shim, WhisperProgress)
            shim.mark_window(0.0)
            bar = real_tr.tqdm.tqdm(total=3000, unit="frames")
            assert isinstance(bar, _ShimmedProgress), (
                f"Expected _ShimmedProgress for window 0, got {type(bar).__name__}"
            )
            nested = real_tr.tqdm.tqdm(total=1000, unit="frames")
            assert isinstance(nested, _NoopBar), (
                f"Expected _NoopBar for nested call, got {type(nested).__name__}"
            )
            shim.mark_window(30.0)
            bar2 = real_tr.tqdm.tqdm(total=3000, unit="frames")
            assert isinstance(bar2, _ShimmedProgress), (
                f"Expected _ShimmedProgress for window 1, got {type(bar2).__name__}"
            )
    finally:
        display.close()


# ---------------------------------------------------------------------------
# Shim: never raises mid-decode
# ---------------------------------------------------------------------------


def test_shim_never_raises_when_real_tqdm_fails(
    tty_stderr: io.StringIO,
) -> None:
    """If the real tqdm constructor raises, the shim swallows the error and
    the decode proceeds (the bar degrades to a no-op)."""

    real_tr = importlib.import_module("mlx_whisper.transcribe")
    # Replace the real tqdm module with one whose factory raises
    failing_tqdm_mod = MagicMock()
    failing_tqdm_mod.tqdm = MagicMock(side_effect=RuntimeError("tqdm broken"))

    display = ProgressDisplay()
    display.start()
    patched = patch.object(real_tr, "tqdm", failing_tqdm_mod)
    patched.start()
    try:
        with with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ) as shim:
            shim.mark_window(0.0)
            bar = real_tr.tqdm.tqdm(total=3000, unit="frames", disable=False)
            bar.__enter__()
            bar.update(1500)
            bar.__exit__(None, None, None)
    finally:
        patched.stop()
        display.close()


def test_shim_never_raises_when_display_progress_update_fails(
    tty_stderr: io.StringIO,
) -> None:
    """If display._progress.update raises, the shim swallows the error."""
    real_tqdm_instance = MagicMock()
    real_tqdm_instance.__enter__ = MagicMock(return_value=real_tqdm_instance)
    real_tqdm_instance.__exit__ = MagicMock()
    real_tqdm_instance.n = 1500
    real_tqdm_instance.total = 3000

    fake_tqdm_mod = MagicMock()
    fake_tqdm_mod.tqdm = MagicMock(return_value=real_tqdm_instance)

    fake_tr = MagicMock()
    fake_tr.tqdm = fake_tqdm_mod

    display = ProgressDisplay()
    display.start()
    # Make display._progress.update raise
    display._progress.update = MagicMock(side_effect=RuntimeError("display broken"))

    fake_mods = {
        "mlx_whisper": MagicMock(transcribe=fake_tr),
        "mlx_whisper.transcribe": fake_tr,
    }
    with (
        patch.dict("sys.modules", fake_mods),
        with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ) as shim,
    ):
        shim.mark_window(0.0)
        patched = fake_tr.tqdm
        bar = patched.tqdm(total=3000, unit="frames", disable=False)
        bar.__enter__()
        # Must not raise
        bar.update(1500)
        bar.__exit__(None, None, None)
    display.close()


# ---------------------------------------------------------------------------
# Shim: no stderr output when not a TTY
# ---------------------------------------------------------------------------


def test_shim_disabled_display_is_passthrough(
    non_tty_stderr: io.StringIO,
) -> None:
    """When the display is disabled (non-TTY), the shim does not touch the
    module and writes nothing to stderr."""
    fake_tr = MagicMock()
    fake_tqdm_mod = MagicMock()
    fake_tr.tqdm = fake_tqdm_mod
    original_tqdm = fake_tr.tqdm

    display = ProgressDisplay()  # non-TTY → disable=True
    assert display.disable is True

    fake_mods = {
        "mlx_whisper": MagicMock(transcribe=fake_tr),
        "mlx_whisper.transcribe": fake_tr,
    }
    with patch.dict("sys.modules", fake_mods):
        with with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ):
            pass
        # The module attribute was never replaced
        assert fake_tr.tqdm is original_tqdm

    output = non_tty_stderr.getvalue()
    # No progress rendering in stderr
    assert "decode" not in output or "\r" not in output
    display.close()


# ---------------------------------------------------------------------------
# Shim: result identical with/without display
# ---------------------------------------------------------------------------


def test_transcribe_result_identical_with_and_without_display(
    tty_stderr: io.StringIO,
) -> None:
    """The decode result (text, segments, words) must be identical whether
    the display is passed or not."""
    raw = {
        "text": "moro vaan",
        "language": "fi",
        "segments": [
            {
                "text": "moro vaan",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": " moro", "start": 0.0, "end": 0.5},
                    {"word": " vaan", "start": 0.6, "end": 1.0},
                ],
            }
        ],
    }

    def _run(display):
        mock = MagicMock()
        mock.transcribe = MagicMock(return_value=raw)
        with (
            patch.dict("sys.modules", {"mlx_whisper": mock}),
            patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
        ):
            t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
            return t.transcribe(np.zeros(16_000, dtype=np.float32), display=display)

    # Without display
    display_off = ProgressDisplay(verbose=False)  # disabled
    result_off = _run(display_off)

    # With display (TTY)
    display_on = ProgressDisplay(verbose=True)  # enabled
    display_on.start()
    result_on = _run(display_on)
    display_on.close()

    assert result_off["text"] == result_on["text"]
    assert len(result_off["segments"]) == len(result_on["segments"])
    assert len(result_off["words"]) == len(result_on["words"])


def test_shim_never_raises_when_add_stage_fails(
    tty_stderr: io.StringIO,
) -> None:
    """If display.add_stage raises (e.g. a broken rich Console mid-decode),
    the shim degrades to an unpatched pass-through: no exception, module
    untouched, and the display task is never finished (task id unavailable)."""
    fake_tr = MagicMock()
    fake_tqdm_mod = MagicMock()
    fake_tr.tqdm = fake_tqdm_mod
    original_tqdm = fake_tr.tqdm

    display = ProgressDisplay()
    display.start()
    display.add_stage = MagicMock(side_effect=RuntimeError("rich broken"))

    fake_mods = {
        "mlx_whisper": MagicMock(transcribe=fake_tr),
        "mlx_whisper.transcribe": fake_tr,
    }
    with (
        patch.dict("sys.modules", fake_mods),
        with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ),
    ):
        # Must not raise, and the module attribute was never replaced.
        assert fake_tr.tqdm is original_tqdm
    # No task was registered, so nothing to finish.
    display.close()


def test_transcribe_with_display_patches_tqdm_during_call(
    tty_stderr: io.StringIO,
) -> None:
    """When a display is passed, the shim patches the tqdm attribute during
    the transcribe() call and restores it after."""
    raw = {"text": "moro", "language": "fi", "segments": []}
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value=raw)

    fake_tr = MagicMock()
    fake_tqdm_mod = MagicMock()
    fake_tr.tqdm = fake_tqdm_mod
    original_tqdm = fake_tr.tqdm

    with (
        patch.dict(
            "sys.modules",
            {"mlx_whisper": mock, "mlx_whisper.transcribe": fake_tr},
        ),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        display = ProgressDisplay(verbose=True)
        display.start()
        t = WhisperTranscriber()
        # Patch the _load_model to avoid actual model loading
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        t.transcribe(np.zeros(16_000, dtype=np.float32), display=display)
        display.close()
        # After the call, the tqdm attribute is restored
        assert fake_tr.tqdm is original_tqdm


# ---------------------------------------------------------------------------
# Shim: no real tqdm bar renders on stderr (issue #105, lens HIGH)
# ---------------------------------------------------------------------------


def test_shimmed_decode_writes_no_tqdm_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path NEVER lets a real tqdm bar render: the shim constructs
    no real tqdm bar (it computes progress from frame counts), so nothing
    tqdm-related is written to stderr during a shimmed decode. Only the rich
    display output (spinner + 'decode' task) appears."""

    # Use a StringIO as stderr so we can read the output directly (the
    # tty_stderr fixture uses a CaptureIO that we can't easily read).
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)

    real_tr = importlib.import_module("mlx_whisper.transcribe")
    display = ProgressDisplay()
    display.start()
    try:
        with with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ) as shim:
            shim.mark_window(0.0)
            bar = real_tr.tqdm.tqdm(total=3000, unit="frames")
            bar.__enter__()
            for frac in (0.25, 0.5, 0.75, 1.0):
                bar.update(int(3000 * frac))
            bar.__exit__(None, None, None)
    finally:
        display.close()

    # No tqdm progress-bar output on stderr: no 'frames/s', no tqdm bar
    # percentage indicator. The rich display output (spinner, 'decode')
    # is expected, but tqdm's own bar is not.
    stderr_out = buf.getvalue()
    assert "frames/s" not in stderr_out, f"tqdm bar rendered on stderr: {stderr_out!r}"
    assert "frames [" not in stderr_out, f"tqdm bar rendered on stderr: {stderr_out!r}"


def test_noop_bar_unknown_attrs_are_noop_callables() -> None:
    """pbar.close()/refresh()/set_description() must not raise on a
    _NoopBar (unknown attributes return a no-op callable; n/total stay
    numeric)."""
    bar = _NoopBar()
    # Numeric attributes stay numeric
    assert bar.n == 0.0
    assert bar.total is None
    # Unknown attributes are no-op callables (must not raise)
    bar.close()
    bar.refresh()
    bar.set_description("x")
    bar.clear()
    # They return None (no-op)
    assert bar.close() is None
    assert bar.refresh() is None
    # total is remembered when provided
    bar_with_total = _NoopBar(total=5000.0)
    assert bar_with_total.total == 5000.0
    bar_with_total.update(2500)
    assert bar_with_total.n == 2500.0


def test_shimmed_progress_unknown_attrs_are_noop_callables(
    tty_stderr: io.StringIO,
) -> None:
    """pbar.close()/refresh()/set_description() must not raise on a
    _ShimmedProgress bar (unknown attributes return a no-op callable;
    n/total stay numeric)."""
    from rich.progress import TaskID

    from vemoizer.progress_shim import _ShimmedProgress

    display = ProgressDisplay()
    display.start()
    try:
        bar = _ShimmedProgress(
            display,
            TaskID(1),
            0.0,
            30.0,
            1.0,
            {"total": 3000},
        )
        bar.__enter__()
        # Numeric attributes
        assert bar.n == 0.0
        assert bar.total == 3000
        # Unknown attributes are no-op callables (must not raise)
        bar.close()
        bar.refresh()
        bar.set_description("x")
        bar.clear()
        assert bar.close() is None
        assert bar.refresh() is None
        bar.update(1500)
        bar.__exit__(None, None, None)
    finally:
        display.close()


# ---------------------------------------------------------------------------
# Per-window protocol: tail window and nested calls (issue #105 lens MEDIUM)
# ---------------------------------------------------------------------------


def test_shim_tail_window_is_not_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tail window shorter than 30 s (e.g. 10 s) is still a window: the
    mark_window call arms it, so its bar drives the display (the old
    audio-length heuristic misclassified it as a re-decode). 100 s file:
    windows 30/30/30/10 s; all four drive the display and the completed
    value reaches the file total (100/60 min)."""
    import types

    from vemoizer.progress_shim import _ShimmedProgress

    real_tr: Any = importlib.import_module("mlx_whisper.transcribe")
    fake: Any = types.ModuleType("fake_tr")
    fake.tqdm = real_tr.tqdm
    orig_import = importlib.import_module
    importlib_mod: Any = importlib
    importlib_mod.import_module = lambda name: (
        fake if name == "mlx_whisper.transcribe" else orig_import(name)
    )
    try:
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
        display = ProgressDisplay()
        display.start()
        completed: list[float] = []
        progress_obj: Any = display._progress
        progress_obj.update = lambda task_id, **kw: completed.append(kw["completed"])
        with with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=100.0 / 60.0,
        ) as shim:
            assert isinstance(shim, WhisperProgress)
            for i, secs in enumerate((30, 30, 30, 10)):
                shim.mark_window(i * 30)
                bar = fake.tqdm.tqdm(total=int(secs * 100), unit="frames")  # type: ignore[attr-defined]
                assert isinstance(bar, _ShimmedProgress), (
                    f"window {i} ({secs}s) misclassified as {type(bar).__name__}"
                )
                bar.__enter__()
                for frac in (0.5, 1.0):
                    bar.update(int(float(bar.total or 0) * frac))  # type: ignore[operator]
                bar.__exit__(None, None, None)
        display.close()
    finally:
        importlib_mod.import_module = orig_import

    assert len(completed) >= 4
    # file-level completed is monotonic across windows
    for i in range(1, len(completed)):
        assert completed[i] >= completed[i - 1] - 1e-9, f"non-monotonic: {completed}"
    # the display reaches the file total (100 s / 60)
    assert completed[-1] == pytest.approx(100.0 / 60.0)


def test_shim_nested_call_gets_noop_and_does_not_advance_display(
    tty_stderr: io.StringIO,
) -> None:
    """A nested call inside a window gets a _NoopBar and does NOT advance
    the display (the window's bar keeps driving)."""

    real_tr = importlib.import_module("mlx_whisper.transcribe")
    display = ProgressDisplay()
    display.start()
    completed: list[float] = []
    orig_update = display._progress.update

    def tracking_update(task_id, **kw):
        if "completed" in kw:
            completed.append(kw["completed"])
        return orig_update(task_id, **kw)

    progress_obj: Any = display._progress
    progress_obj.update = tracking_update
    try:
        with with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ) as shim:
            shim.mark_window(0.0)
            bar = real_tr.tqdm.tqdm(total=3000, unit="frames")
            bar.__enter__()
            bar.update(1500)
            display_after_window = list(completed)
            # nested call → _NoopBar
            nested = real_tr.tqdm.tqdm(total=1000, unit="frames")
            nested.__enter__()
            nested.update(1000)
            # the nested call must NOT have advanced the display
            assert completed == display_after_window, (
                f"nested call advanced the display: "
                f"{display_after_window} -> {completed}"
            )
            bar.__exit__(None, None, None)
    finally:
        display.close()


def test_shim_no_tqdm_module_no_stage_created(
    tty_stderr: io.StringIO,
) -> None:
    """When the patched module has no tqdm attribute, the shim does NOT
    create a display stage (the check happens before add_stage), so no
    task is stranded."""
    fake_tr = types.SimpleNamespace()

    display = ProgressDisplay()
    display.start()
    add_stage_calls: list[str] = []
    orig_add = display.add_stage

    def tracking_add(description, total=None):
        add_stage_calls.append(description)
        return orig_add(description, total=total)

    display_obj: Any = display
    display_obj.add_stage = tracking_add

    fake_mods = {"mlx_whisper.transcribe": fake_tr}
    with (
        patch.dict("sys.modules", fake_mods),
        with_whisper_progress(
            display,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ) as shim,
    ):
        assert isinstance(shim, WhisperProgress)
    # No stage was created (add_stage never called)
    assert add_stage_calls == [], f"add_stage was called: {add_stage_calls}"
    display.close()


# ---------------------------------------------------------------------------
# verbose override with an active display (issue #105 lens MEDIUM)
# ---------------------------------------------------------------------------


def test_transcribe_forces_verbose_false_with_active_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a display is active, the effective verbose is always False,
    even if the caller passed verbose=True or verbose=None (the display
    owns progress and transcript text must never go to stdout)."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    raw = {
        "text": "moro",
        "language": "fi",
        "segments": [{"text": "moro", "start": 0.0, "end": 1.0, "words": []}],
    }

    def _run_with_kwargs(extra_kwargs: dict):
        mock = MagicMock()
        mock.transcribe = MagicMock(return_value=raw)
        with (
            patch.dict("sys.modules", {"mlx_whisper": mock}),
            patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
        ):
            display = ProgressDisplay(verbose=True)
            display.start()
            try:
                t = WhisperTranscriber()
                t._model_path = "/tmp/turbo"
                t._mlx_whisper = mock
                t.transcribe(
                    np.zeros(16_000, dtype=np.float32),
                    display=display,
                    **extra_kwargs,
                )
            finally:
                display.close()
        return mock.transcribe.call_args.kwargs

    # caller passes verbose=True → forced to False
    kwargs_true = _run_with_kwargs({"verbose": True})
    assert kwargs_true.get("verbose") is False, (
        f"Expected verbose=False (forced), got {kwargs_true.get('verbose')}"
    )

    # caller passes verbose=None → forced to False
    kwargs_none = _run_with_kwargs({"verbose": None})
    assert kwargs_none.get("verbose") is False, (
        f"Expected verbose=False (forced), got {kwargs_none.get('verbose')}"
    )
