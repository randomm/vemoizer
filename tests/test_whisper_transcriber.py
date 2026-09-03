"""WhisperTranscriber — turbo decode A for the meeting profile (issue #71).

mlx_whisper is mocked throughout: no downloads, no GPU. The transcriber
feeds the WHOLE recording to one transcribe() call (mlx-whisper windows
internally) and surfaces words/segments/language; per-VAD-slice records
for dispute detection are derived from the word timestamps.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vemoizer.whisper_transcriber import (
    MODEL_ID,
    MODEL_REVISION,
    WhisperTranscriber,
    slice_records_from_words,
)


def _raw(segments):
    return {
        "text": " ".join(s["text"] for s in segments),
        "language": "fi",
        "segments": segments,
    }


def _seg(text, words):
    return {
        "text": text,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


def _mock_whisper(raw):
    m = MagicMock()
    m.transcribe = MagicMock(return_value=raw)
    return m


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def test_model_is_revision_pinned_turbo() -> None:
    assert MODEL_ID == "mlx-community/whisper-large-v3-turbo"
    assert len(MODEL_REVISION) == 40


def test_transcribe_is_one_call_over_the_whole_recording() -> None:
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
    mock = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        result = t.transcribe(_audio(120.0))

    assert mock.transcribe.call_count == 1  # never per-slice
    kwargs = mock.transcribe.call_args.kwargs
    assert kwargs["path_or_hf_repo"] == "/tmp/turbo"
    assert kwargs["word_timestamps"] is True
    assert kwargs["temperature"] == 0.0
    assert kwargs["condition_on_previous_text"] is False
    assert result["text"] == "moro vaan"
    assert result["language"] == "fi"
    assert [w["word"] for w in result["words"]] == ["moro", "vaan"]
    assert result["words"][0]["start"] == 0.0
    assert result["segments"][0]["text"] == "moro vaan"


def test_transcribe_empty_audio_short_circuits() -> None:
    mock = _mock_whisper({})
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        result = WhisperTranscriber().transcribe(np.zeros(0, dtype=np.float32))
    assert result["text"] == ""
    assert mock.transcribe.call_count == 0


def test_load_failure_latches_and_raises() -> None:
    with patch(
        "huggingface_hub.snapshot_download", side_effect=RuntimeError("offline")
    ) as dl:
        t = WhisperTranscriber()
        with pytest.raises(RuntimeError):
            t.transcribe(_audio(1.0))
        with pytest.raises(RuntimeError):
            t.transcribe(_audio(1.0))
    assert dl.call_count == 1  # latched


# -- slice_records_from_words --------------------------------------------


def test_words_map_into_vad_slice_bounds() -> None:
    words = [
        {"word": "eka", "start": 0.5, "end": 0.9},
        {"word": "toka", "start": 1.2, "end": 1.6},
        {"word": "kolmas", "start": 10.1, "end": 10.6},
    ]
    slices = [(0, _audio(2.0)), (int(16_000 * 10.0), _audio(1.0))]
    records = slice_records_from_words(words, slices, language="fi")
    assert [r["index"] for r in records] == [0, 1]
    assert records[0]["text"] == "eka toka"
    assert records[0]["start_s"] == 0.0
    assert records[0]["end_s"] == 2.0
    assert records[1]["text"] == "kolmas"
    assert records[1]["start_s"] == 10.0
    assert all(r["language"] == "fi" for r in records)


def test_slice_with_no_words_yields_empty_text_record() -> None:
    """A silent slice still gets a record: 'whisper heard nothing here' is
    a signal the dispute stage must see (vs the slice being missing)."""
    records = slice_records_from_words([], [(0, _audio(2.0))], language=None)
    assert records[0]["text"] == ""
    assert "language" not in records[0]
