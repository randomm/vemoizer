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
from vemoizer.whisper_windows import MIN_VAD_OVERLAP_S, _window_has_speech

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


def _run_transcribe(
    audio,
    side_effect,
    *,
    prompt: str | None = None,
    vad_slices=None,
):
    """Drive ``WhisperTranscriber.transcribe`` with a mocked mlx_whisper.

    ``vad_slices`` (``(offset, seconds)`` pairs) is threaded to the
    transcriber as the seam's VAD availability signal (issue #152 FIX 1);
    ``None`` means the caller did not ask for the seam to be exercised.
    """
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=side_effect)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt=prompt)
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        if vad_slices is not None:
            result = t.transcribe(audio, vad_slices=vad_slices)
        else:
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


def test_hiss_only_window_is_not_retry_or_loss(caplog) -> None:
    """FIX 1: a hiss-only window (peak ~5e-3, room noise) with an empty
    decode gets NO retry, is NOT in lost_windows, and logs no warning —
    ordinary pauses must not produce 'puhetta, ei tekstiä' report lines."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    hiss = _hiss_audio(30.0, 5e-3, seed=7)
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(hiss, [empty_raw, empty_raw])
    assert mock.transcribe.call_count == 1  # no retry on a no-speech window
    assert "lost_windows" not in result
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert warnings == []


def test_hiss_louder_window_is_not_retry_or_loss(caplog) -> None:
    """FIX 1: even hiss at 2e-2 peak (louder room noise) is below the
    fallback RMS gate: no retry, no loss, no warning."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    # Gaussian noise at 5e-3 scale: peak ~2e-2, RMS ~5e-3 (-46 dBFS).
    hiss = _hiss_audio(30.0, 5e-3, seed=8)
    assert float(np.abs(hiss).max()) > 1.5e-2
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(hiss, [empty_raw, empty_raw])
    assert mock.transcribe.call_count == 1
    assert "lost_windows" not in result
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert warnings == []


def _hiss_audio(seconds: float, scale: float, seed: int = 0) -> np.ndarray:
    """Hiss-only audio (Gaussian noise at *scale* amplitude)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * 16_000)) * scale).astype(np.float32)


def _hiss_with_speech_speech_seconds(seconds: float) -> np.ndarray:
    """A hiss-only window plus a 3 s speech-like sine burst at the start,
    matching the VAD slice below."""
    rng = np.random.default_rng(9)
    audio = (rng.standard_normal(30 * 16_000) * 5e-3).astype(np.float32)
    t = np.linspace(0, seconds, int(seconds * 16_000), endpoint=False)
    burst = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio[: len(burst)] += burst
    return audio


def test_vad_overlap_window_empty_decode_retries_keeps_text(caplog) -> None:
    """FIX 1: a window the VAD slices flag as speech (>0.5 s overlap) with
    an empty decode retries once, warning logged, and the retry's text is
    kept."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    good_raw = _raw(
        [
            _seg("tässä oli puhetta", [{"word": " tässä", "start": 0.0, "end": 0.3}]),
        ]
    )
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _hiss_with_speech_speech_seconds(3.0),
            [empty_raw, good_raw],
            prompt="Sanasto: Flagship.",
            vad_slices=[(0, int(3.0 * 16_000))],  # 3 s slice (sample units)
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


def test_vad_overlap_window_still_empty_after_retry_is_lost(caplog) -> None:
    """FIX 1: a window the VAD slices flag as speech that is still empty
    after the prompt-free retry is recorded in lost_windows with a warning."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _hiss_with_speech_speech_seconds(3.0),
            [empty_raw, empty_raw],
            prompt="Sanasto: Flagship.",
            vad_slices=[(0, int(3.0 * 16_000))],  # 3 s slice (sample units)
        )
    assert mock.transcribe.call_count == 2
    assert result["lost_windows"] == [(0.0, 30.0)]
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_vad_overlap_below_minimum_is_not_speech(caplog) -> None:
    """FIX 1: a VAD slice that overlaps a window by only 0.4 s (below
    MIN_VAD_OVERLAP_S) is a VAD artifact, not speech: no retry, no loss."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    # Pure hiss (no speech burst) so the fallback RMS gate also says silent.
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _hiss_audio(30.0, 5e-3, seed=7),
            [empty_raw, empty_raw],
            vad_slices=[(0, int(0.4 * 16_000))],  # 0.4 s into window 0
        )
    assert mock.transcribe.call_count == 1
    assert "lost_windows" not in result
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert warnings == []


def test_vad_slice_spanning_two_windows_counts_both(caplog) -> None:
    """FIX 1: one VAD slice spanning the boundary of windows 0 and 1
    counts as speech for BOTH windows (overlap ≥ 0.5 s each): both retry."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _speech_audio(60.0),
            [empty_raw, empty_raw, empty_raw, empty_raw],  # 2 decodes + 2 retries
            vad_slices=[(int(28.0 * 16_000), int(32.0 * 16_000))],  # 2 s into each
        )
    assert mock.transcribe.call_count == 4
    assert sorted(result["lost_windows"]) == [(0.0, 30.0), (30.0, 60.0)]


def test_vad_zero_slices_falls_back_to_rms_gate(caplog) -> None:
    """FIX 1: an empty slices list means VAD is unavailable: the fallback
    frame-RMS gate applies. Hiss alone -> silent (no retry, no loss); the
    same audio with a speech-like burst -> speech (retry, lost when empty)."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    rng = np.random.default_rng(10)
    hiss = (rng.standard_normal(30 * 16_000) * 5e-3).astype(np.float32)
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(hiss, [empty_raw], vad_slices=())
    assert mock.transcribe.call_count == 1  # hiss only: silent under the RMS gate
    assert "lost_windows" not in result
    speech = _hiss_with_speech_speech_seconds(3.0)
    with caplog.at_level(logging.WARNING):
        mock2, result2 = _run_transcribe(speech, [empty_raw, empty_raw], vad_slices=())
    assert mock2.transcribe.call_count == 2  # speech burst: retried
    assert result2["lost_windows"] == [(0.0, 30.0)]


def test_retry_zero_segments_logs_warning(caplog) -> None:
    """FIX 4: a retry that returns 0 segments (not None) logs a warning
    naming the window index and offset, and records the window as lost."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _speech_audio(10.0), [empty_raw, empty_raw], prompt="Sanasto: Flagship."
        )
    assert mock.transcribe.call_count == 2
    assert result["lost_windows"] == [(0.0, 30.0)]
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "returned 0 segments" in r.getMessage()
        and "recording as lost" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "window 0" in warnings[0].getMessage()
    assert "offset 0s" in warnings[0].getMessage()


def test_fallback_rms_gate_levels() -> None:
    """FIX 1 fallback gate levels: zeros -> silent; hiss at 5e-3 -> silent;
    speech-like modulated noise at 0.1 peak -> speech; quiet-but-real
    speech-like signal at ~-35 dBFS RMS -> speech."""
    assert _window_has_speech(np.zeros(30 * 16_000, dtype=np.float32)) is False
    rng = np.random.default_rng(11)
    hiss = (rng.standard_normal(30 * 16_000) * 5e-3).astype(np.float32)
    assert _window_has_speech(hiss) is False
    t = np.linspace(0, 30.0, 30 * 16_000, endpoint=False)
    modulated = 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)) * np.sin(200.0 * t)
    assert _window_has_speech(modulated.astype(np.float32)) is True
    quiet = (0.015 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)  # ~-28 dBFS
    assert _window_has_speech(quiet) is True
    assert _window_has_speech(np.zeros(0, dtype=np.float32)) is False


def test_retry_keeps_configured_language_exactly() -> None:
    """Declined-but-documented (invariant 3): the prompt-free retry passes
    ``language`` exactly as configured — a None (per-window detection) stays
    None; a run-level pin stays pinned. No re-detection, no change."""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    good_raw = _raw([_seg("moro", [{"word": " moro", "start": 0.0, "end": 0.3}])])
    for language, expected in ((None, None), ("fi", "fi")):
        mock = MagicMock()
        mock.transcribe = MagicMock(side_effect=[empty_raw, good_raw])
        with (
            patch.dict("sys.modules", {"mlx_whisper": mock}),
            patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
        ):
            t = WhisperTranscriber(
                language=language, initial_prompt="Sanasto: Flagship."
            )
            t._model_path = "/tmp/turbo"
            t._mlx_whisper = mock
            t.transcribe(_speech_audio(10.0))
        retry_kwargs = mock.transcribe.call_args_list[1].kwargs
        assert retry_kwargs["language"] is expected, language


def test_window_has_speech_min_vad_overlap_constant() -> None:
    """The VAD overlap gate is a documented positive constant (seconds)."""
    assert MIN_VAD_OVERLAP_S > 0
    assert MIN_VAD_OVERLAP_S <= 1.0
