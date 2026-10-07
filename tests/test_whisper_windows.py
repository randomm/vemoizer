"""Tests for per-window language counting and summary (issue #147).

These tests exercise :func:`vemoizer.whisper_windows.process_window_raws`
via the :class:`WhisperTranscriber` decode path, verifying that the
per-window detected languages are aggregated into a ``language_summary``
string and that the summary is correct for mixed fi/en, all-fi, and
zero-window cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from vemoizer.whisper_transcriber import WhisperTranscriber


def _seg(text: str, words: list[dict]) -> dict:
    return {
        "text": text,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


def _raw_lang(language: str, text: str = "moro") -> dict:
    return {
        "text": text,
        "language": language,
        "segments": [_seg(text, [{"word": " " + text, "start": 0.0, "end": 0.5}])],
    }


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def _mock_whisper(raw: dict) -> MagicMock:
    m = MagicMock()
    m.transcribe = MagicMock(return_value=raw)
    return m


# -- per-window language counting + summary (issue #147) ----------------------


def test_language_summary_mixed_languages() -> None:
    """A mixed fi/en run produces the correct per-window distribution."""
    raws = [_raw_lang("fi") for _ in range(29)] + [_raw_lang("en")]
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=raws)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(_audio(30 * 30))
    assert mock.transcribe.call_count == 30
    assert "language" not in result  # disagreement → no single language
    assert result["language_summary"] == "fi 29/30, en 1/30"


def test_language_summary_single_language() -> None:
    """A single-language run sets both "language" and "language_summary"."""
    raws = [_raw_lang("fi") for _ in range(3)]
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=raws)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(_audio(90.0))
    assert result.get("language") == "fi"
    assert result["language_summary"] == "fi 3/3"


def test_language_summary_absent_on_empty_audio() -> None:
    """Zero windows (empty audio) → no "language", no "language_summary"."""
    with (
        patch.dict("sys.modules", {"mlx_whisper": _mock_whisper({})}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        result = WhisperTranscriber().transcribe(np.zeros(0, dtype=np.float32))
    assert "language" not in result
    assert "language_summary" not in result
    assert result["text"] == ""
