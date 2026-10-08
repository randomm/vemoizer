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
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from _cli_helpers import isolate_home
from _whisper_helpers import _audio, _raw, _seg, _speech_audio
from test_pipeline import _patch_ingest, _patch_preflight_pass, _patch_vad
from typer.testing import CliRunner

import vemoizer.pipeline as vemoizer_pipeline
from vemoizer.vad import SpeechSegment
from vemoizer.whisper_transcriber import WhisperTranscriber
from vemoizer.whisper_windows import (
    MIN_VAD_OVERLAP_S,
    _window_has_speech,
    normalize_lost_windows,
)

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

    ``vad_slices`` (``(start_sample, end_sample)`` sample-index pairs on the
    recording timeline) is threaded to the transcriber as the seam's VAD
    availability signal (issue #152 FIX 1): ``None`` means "no usable VAD
    information — use the frame-RMS fallback", while a (possibly empty)
    list means "the VAD ran and these are its speech spans — trust them".
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


def test_vad_empty_list_means_no_speech_nowhere(caplog) -> None:
    """FIX 1 (explicit signal): an EMPTY vad_slices list means "the VAD ran
    and found zero speech spans" — every window is non-speech regardless of
    energy: no retry, no loss, even for a loud speech-like signal. (The
    "no VAD information at all" case is ``vad_slices=None``, which the
    ``_run_transcribe`` default exercises throughout this module.)"""
    empty_raw = {"text": "", "language": "fi", "segments": []}
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            _hiss_with_speech_speech_seconds(30.0), [empty_raw], vad_slices=[]
        )
    assert mock.transcribe.call_count == 1  # VAD ran, zero spans: no retry
    assert "lost_windows" not in result
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert warnings == []


def test_all_speech_recording_quiet_signal_is_retried(caplog) -> None:
    """FIX 1 (explicit signal): a genuine short all-speech recording whose
    single VAD slice covers the whole file ((0, len)) with a quiet speech
    signal (~-45 dBFS RMS, BELOW the -40 dBFS fallback floor) and an empty
    decode: the old shape heuristic mistook the full-coverage slice for the
    "VAD found nothing" fallback and the RMS gate wrongly called the quiet
    speech silent (no retry). The explicit VAD signal says speech — the
    window is retried, and the empty retry is recorded as lost."""
    t = np.linspace(0, 30.0, 30 * 16_000, endpoint=False)
    quiet = (0.006 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)  # ~-44.4 dBFS
    n = len(quiet)
    rms = float(np.sqrt((np.square(quiet).astype(np.float64)).mean()))
    assert 0.004 < rms < 0.008  # below the 1e-2 (-40 dBFS) fallback floor
    empty_raw = {"text": "", "language": "fi", "segments": []}
    with caplog.at_level(logging.WARNING):
        mock, result = _run_transcribe(
            quiet,
            [empty_raw, empty_raw],
            prompt="Sanasto: Flagship.",
            vad_slices=[(0, n)],
        )
    assert mock.transcribe.call_count == 2  # VAD says speech: retried
    assert result["lost_windows"] == [(0.0, 30.0)]
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "returned no segments" in r.getMessage()
    ]
    assert len(warnings) == 1


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
    quiet = (0.015 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)  # ~-39.5 dBFS
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


# -- numpy-scalar acceptance (issue #152, fix pass 3) ----------------------


def test_normalize_lost_windows_accepts_numpy_numeric_scalars() -> None:
    """numpy numeric scalars (np.float32/np.int64/np.float64) are real
    numerics and are accepted, exactly as with plain int/float; bool (a
    bool subclass of int, still excluded) and the existing garbage rules
    (NaN, negative, start > end) are unchanged."""
    for a, b in (
        (np.float32(1.5), np.float64(30.0)),
        (np.int64(2), np.int64(30)),
        (np.float64(0.0), np.float32(30.0)),
    ):
        assert normalize_lost_windows([(a, b)]) == [(float(a), float(b))]
    # bool is still excluded even in numpy form.
    assert normalize_lost_windows([(np.bool_(True), np.int32(2))]) == []
    # Garbage rules unchanged under numpy scalars.
    assert normalize_lost_windows([(np.float64(float("nan")), np.float64(2.0))]) == []
    assert normalize_lost_windows([(np.float64(-1.0), np.float64(2.0))]) == []
    assert normalize_lost_windows([(np.float64(5.0), np.float64(4.0))]) == []


# -- pipeline-level seam: the explicit signal decode_meeting threads -------


def _spy_decode_meeting(monkeypatch, seen: dict) -> None:
    """Record every ``decode_meeting`` call's kwargs (``vad_slices``
    threaded via the ``transcribe`` kwarg the seam adds) and return a
    valid meeting result so the run never touches decode B."""
    import vemoizer.pipeline as pipeline_module

    def fake_decode_meeting(audio, slices, initial_prompt=None, **kwargs):
        seen["calls"].append(kwargs)
        return {
            "text": "hei maailma",
            "words": [{"word": "hei", "start": 0.0, "end": 0.4}],
            "segments": [{"start": 0.0, "end": 1.0, "text": "hei maailma"}],
            "slices": [],
        }

    monkeypatch.setattr(pipeline_module, "decode_meeting", fake_decode_meeting)


def _run_real_meeting_pipeline(
    monkeypatch, tmp_path: Path, *, extra_args=None, vad_patch=None
):
    """Invoke ``meeting`` with the REAL ``transcribe_file`` body.

    The heavy stages are faked at the module seams as
    ``test_meeting_language`` does (preflight forced green, ingest and VAD
    patched); only ``decode_meeting`` is the spy, so the REAL
    ``_speech_slices`` decides the three VAD cases. ``vad_patch`` is an
    optional callable applied AFTER the default VAD patch so it wins.
    Returns the spy's recorded kwargs.
    """
    import vemoizer.pipeline as pipeline_module

    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    if vad_patch is not None:
        vad_patch(monkeypatch)
    # Import the HF token helper BEFORE the test's logging snapshot (the
    # autouse guard in conftest) so a StreamHandler the import attaches
    # predates the snapshot and cannot look like a leak.
    from huggingface_hub import get_token  # noqa: F401

    import vemoizer.preflight as preflight_module

    monkeypatch.setattr(preflight_module, "hf_token_present", lambda: True)
    seen: dict = {"calls": []}
    _spy_decode_meeting(monkeypatch, seen)

    import vemoizer.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "pcm_duration_seconds", lambda path, **kw: 2.0)

    def fake_diarize(audio, speakers=None):
        return [(0.0, 2.0, "SPEAKER_00")]

    monkeypatch.setattr(pipeline_module, "run_diarization_stage", fake_diarize)
    _config_no_llm(tmp_path)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    args = ["meeting", "a.m4a"] + (extra_args or [])
    from vemoizer.cli import app

    runner = CliRunner()
    result = runner.invoke(app, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return seen


def _config_no_llm(tmp_path: Path) -> None:
    """Home-layer config with a valid ``[llm]`` section pointing at an
    unroutable host: the layered search accepts it and every LLM call
    fails open (no network needed)."""
    home = tmp_path / "home"
    (home / ".vemoizer").mkdir(parents=True, exist_ok=True)
    (home / ".vemoizer" / "config.toml").write_text(
        "[llm]\n"
        'base_url = "https://llm.invalid/v1"\n'
        'model = "no-network"\n'
        'api_key_env = "K"\n'
        "timeout_seconds = 0.05\n",
        encoding="utf-8",
    )


def test_pipeline_passes_real_vad_slices_to_decode_meeting(
    monkeypatch, tmp_path: Path
) -> None:
    """Case 1 (VAD found real speech, even full-file coverage): the spy
    decode_meeting receives the explicit (start, end) sample pairs —
    a genuine single full-recording slice is NOT mistaken for the
    "VAD found nothing" fallback."""
    seen = _run_real_meeting_pipeline(
        monkeypatch,
        tmp_path,
        vad_patch=lambda m: m.setattr(
            vemoizer_pipeline,
            "vad_segments",
            lambda a, mod: [SpeechSegment(160, 16000)],
        ),
    )
    assert len(seen["calls"]) == 1
    assert seen["calls"][0]["vad_slices"] == [(160, 16000)]


def test_pipeline_passes_empty_slice_list_when_vad_finds_no_speech(
    monkeypatch, tmp_path: Path
) -> None:
    """Case 2 (VAD ran, zero speech spans): the explicit signal is the
    EMPTY LIST, not None — every window is non-speech, no RMS fallback."""
    seen = _run_real_meeting_pipeline(
        monkeypatch,
        tmp_path,
        vad_patch=lambda m: m.setattr(
            vemoizer_pipeline, "vad_segments", lambda a, mod: []
        ),
    )
    assert len(seen["calls"]) == 1
    assert seen["calls"][0]["vad_slices"] is None


def test_pipeline_passes_none_when_vad_unavailable(monkeypatch, tmp_path: Path) -> None:
    """Case 3 (VAD unavailable — the full-file slice the pipeline
    substitutes is NOT VAD output): the explicit signal is None, so the
    frame-RMS fallback gate decides inside the transcriber."""

    def broken_vad(a, m):
        raise RuntimeError("onnx session init failed")

    seen = _run_real_meeting_pipeline(
        monkeypatch,
        tmp_path,
        vad_patch=lambda m: m.setattr(vemoizer_pipeline, "vad_segments", broken_vad),
    )
    assert len(seen["calls"]) == 1
    assert seen["calls"][0]["vad_slices"] is None
