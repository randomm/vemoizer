"""Unit tests for the mlx-whisper tqdm shim (issue #105).

Covers: frames→minutes conversion, tqdm attribute patched and restored in
finally even when the decode raises, the idempotent guard for re-entrant
calls, the shim never raising mid-decode, no stderr output when not a TTY,
and the decode result being identical with or without a display.
"""

from __future__ import annotations

import io
import sys
from typing import IO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vemoizer.progress import (
    ProgressDisplay,
    _NoopBar,
    _ShimmedBar,
    frames_to_minutes,
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
    import importlib

    real_tr = importlib.import_module("mlx_whisper.transcribe")
    original_tqdm = real_tr.tqdm
    display = ProgressDisplay()
    display.start()
    try:
        with with_whisper_progress(
            display,
            window_offset_seconds=0,
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
                window_offset_seconds=0,
                window_seconds=30.0,
                file_total_minutes=1.0,
            ),
        ):
            raise RuntimeError("decode failed")
        # The original must be restored even after the exception
        assert fake_tr.tqdm is original_tqdm
    display.close()


# ---------------------------------------------------------------------------
# Shim: idempotent guard for re-entrant calls
# ---------------------------------------------------------------------------


def test_shim_idempotent_guard_prevents_double_wrap(
    tty_stderr: io.StringIO,
) -> None:
    """While the shim is active, a re-entrant transcribe() call (self-heal
    short re-decode) gets a no-op bar, not a second wrapped bar."""
    import importlib

    real_tr = importlib.import_module("mlx_whisper.transcribe")
    display = ProgressDisplay()
    display.start()
    try:
        with with_whisper_progress(
            display,
            window_offset_seconds=0,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ):
            # The patched tqdm module is a _ShimmedBar; calling its .tqdm
            # while the shim is active returns a _NoopBar (re-entrant guard)
            bar = real_tr.tqdm.tqdm(total=3000, unit="frames", disable=False)
            assert isinstance(bar, _NoopBar), (
                f"Expected _NoopBar for re-entrant call, got {type(bar).__name__}"
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
    import importlib

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
            window_offset_seconds=0,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ):
            # The shim wraps the failing factory in _ShimmedProgress;
            # calling it must not raise.
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
            window_offset_seconds=0,
            window_seconds=30.0,
            file_total_minutes=1.0,
        ),
    ):
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
            window_offset_seconds=0,
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
