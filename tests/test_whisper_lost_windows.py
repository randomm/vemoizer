"""Lost-window retry for the Whisper meeting decode (issue #152, FIX 1/2).

The prompt-free fail-safe retry of 0-segment windows: a window with speech
energy that decodes to 0 segments is re-decoded once without the glossary
prompt, and a window that is lost (empty/None even after the retry) is
recorded in ``lost_windows``. Helpers shared with ``test_whisper_transcriber``
live in ``_whisper_helpers``.

mlx_whisper is mocked throughout: no downloads, no GPU.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
from _whisper_helpers import _audio, _raw, _seg, _speech_audio

from vemoizer.whisper_transcriber import WhisperTranscriber
from vemoizer.whisper_windows import SILENT_PEAK_THRESHOLD, _window_has_speech

# -- fail-safe retry for 0-segment windows (issue #152) -------------------


def test_zero_segment_window_triggers_prompt_free_retry() -> None:
    """A window that returns 0 segments (and has speech energy) triggers a
    retry without the glossary prompt; the retry's text is kept."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    good_raw = _raw(
        [
            _seg(
                "tässä oli puhetta",
                [
                    {"word": " tässä", "start": 0.0, "end": 0.3},
                    {"word": " oli", "start": 0.4, "end": 0.5},
                    {"word": " puhetta", "start": 0.6, "end": 0.9},
                ],
            )
        ]
    )
    mock = MagicMock()
    # First call (window 0) returns empty; retry (window 0, no prompt) returns good.
    mock.transcribe = MagicMock(side_effect=[empty_raw, good_raw])
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(_speech_audio(10.0))
    # Two calls: initial decode + prompt-free retry.
    assert mock.transcribe.call_count == 2
    # The retry dropped the prompt.
    retry_kwargs = mock.transcribe.call_args_list[1].kwargs
    assert retry_kwargs["initial_prompt"] is None
    # The retry's text is in the result.
    assert "tässä oli puhetta" in result["text"]
    # No lost_windows key (the retry succeeded).
    assert "lost_windows" not in result


def test_silent_zero_segment_window_does_not_retry() -> None:
    """A genuinely silent window (all zeros) that returns 0 segments does NOT
    trigger a retry — there's nothing to decode."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value=empty_raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(_audio(10.0))
    # Only one call (no retry for silent audio).
    assert mock.transcribe.call_count == 1
    assert result["text"] == ""


def test_zero_segment_retry_fails_open() -> None:
    """If the prompt-free retry also returns 0 segments, the window is
    recorded in lost_windows (fail-open: the transcript is incomplete but
    the run continues)."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    mock = MagicMock()
    # Both calls return empty (initial + retry).
    mock.transcribe = MagicMock(return_value=empty_raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(_speech_audio(10.0))
    # Two calls: initial + retry.
    assert mock.transcribe.call_count == 2
    # The window is recorded as lost with (start_s, end_s) time range.
    assert "lost_windows" in result
    assert len(result["lost_windows"]) == 1
    start_s, end_s = result["lost_windows"][0]
    assert start_s == 0.0  # window 0 starts at 0s
    assert end_s == 30.0  # 30 s window


def test_zero_segment_retry_none_records_lost() -> None:
    """A retry that returns None (mlx-whisper's documented failure mode)
    is a real failure for a window with confirmed speech energy: it is
    recorded in lost_windows (never silent), not silently skipped."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[empty_raw, None])
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(_speech_audio(10.0))
    assert mock.transcribe.call_count == 2
    # Fail-open: transcript is incomplete but the run continued; and the
    # loss is observable (a gap is never silent).
    assert "lost_windows" in result
    assert result["lost_windows"] == [(0.0, 30.0)]
    assert result["text"] == ""


def test_no_retry_when_all_windows_have_segments() -> None:
    """When all windows return segments, no retry calls are made."""
    raw = _raw(
        [
            _seg(
                "moro vaan",
                [
                    {"word": " moro", "start": 0.0, "end": 0.5},
                    {"word": " vaan", "start": 0.6, "end": 1.0},
                ],
            )
        ]
    )
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value=raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        t.transcribe(_speech_audio(60.0))
    # Exactly 2 calls (one per 30 s window), no retries.
    assert mock.transcribe.call_count == 2


# -- the silent-window gate (issue #152, FIX 2) ---------------------------


def _run_transcribe(audio, side_effect, *, prompt: str | None = None):
    """Drive ``WhisperTranscriber.transcribe`` with a mocked mlx_whisper."""
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=side_effect)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt=prompt)
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(audio)
    return mock, result


def test_silent_window_empty_decode_is_not_retry_or_loss(caplog) -> None:
    """FIX 2: a genuinely silent window (zeros) with an empty decode gets NO
    retry call, is NOT in lost_windows, and no warning is logged — silence
    is not a lost window."""

    empty_raw = {"text": "", "language": "fi", "segments": []}
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            np.zeros(2 * 30 * 16_000, dtype=np.float32), [empty_raw, empty_raw]
        )
    # 2 windows, each decoded exactly once: no retry on silent audio.
    assert mock.transcribe.call_count == 2
    assert "lost_windows" not in result
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert warnings == []


def test_speech_window_empty_decode_retries_keeps_text(caplog) -> None:
    """FIX 2: a speech window with an empty decode retries once (warning
    logged) and the retry's text is kept."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    good_raw = _raw(
        [
            _seg(
                "tässä oli puhetta",
                [{"word": " tässä", "start": 0.0, "end": 0.3}],
            )
        ]
    )
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _speech_audio(30.0), [empty_raw, good_raw], prompt="Sanasto: Flagship."
        )
    assert mock.transcribe.call_count == 2
    retry_kwargs = mock.transcribe.call_args_list[1].kwargs
    assert retry_kwargs["initial_prompt"] is None
    assert "tässä oli puhetta" in result["text"]
    assert "lost_windows" not in result
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "retrying without glossary prompt" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_speech_window_empty_after_retry_is_lost_with_warning(caplog) -> None:
    """FIX 2: a speech window empty even after the retry ends up in
    lost_windows with a warning (the gap is never silent)."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _speech_audio(30.0), [empty_raw, empty_raw], prompt="Sanasto: Flagship."
        )
    assert mock.transcribe.call_count == 2
    assert result["lost_windows"] == [(0.0, 30.0)]
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_window_has_speech_zeros_noise_and_quiet_real() -> None:
    """The energy gate: zeros (silence) below threshold, noise above, and a
    quiet-but-real level above."""
    assert _window_has_speech(np.zeros(16_000, dtype=np.float32)) is False
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(16_000) * 0.1).astype(np.float32)
    assert _window_has_speech(noise) is True
    quiet = (np.ones(16_000) * 3e-5).astype(np.float32)
    assert _window_has_speech(quiet) is True
    assert _window_has_speech(np.zeros(0, dtype=np.float32)) is False
    assert SILENT_PEAK_THRESHOLD > 0
