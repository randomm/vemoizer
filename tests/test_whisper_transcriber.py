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
    decode_meeting,
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
    assert kwargs["temperature"][0] == 0.0  # deterministic-first ladder
    assert kwargs["condition_on_previous_text"] is True
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


# -- reliability knobs (issue #71 round 2) -------------------------------
#
# Research findings: (a) with condition_on_previous_text=False the
# initial_prompt (glossary) reaches ONLY the first 30s window — verified
# in mlx-whisper 0.4.3 source (prompt_reset_since advances every window);
# (b) the temperature fallback ladder + thresholds is whisper's designed
# anti-loop mechanism, letting conditioning stay on safely; (c) per-
# segment confidence must be kept, not discarded — it feeds flagging.


def test_conditioning_on_with_fallback_ladder() -> None:
    raw = _raw([_seg("moro", [{"word": " moro", "start": 0.0, "end": 0.5}])])
    mock = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        WhisperTranscriber(initial_prompt="Sanasto: Flagship.").transcribe(_audio(60.0))
    kwargs = mock.transcribe.call_args.kwargs
    # Rolling context carries the glossary vocabulary past the first window
    assert kwargs["condition_on_previous_text"] is True
    # ...safely: the paper-validated anti-loop stack
    assert kwargs["temperature"] == (0.0, 0.2, 0.4)
    assert kwargs["compression_ratio_threshold"] == 2.4
    assert kwargs["logprob_threshold"] == -1.0
    assert kwargs["no_speech_threshold"] == 0.6
    assert kwargs["hallucination_silence_threshold"] == 2.0


def test_segment_confidence_is_kept() -> None:
    seg = _seg("moro vaan", [{"word": " moro", "start": 0.0, "end": 0.5}])
    seg["avg_logprob"] = -0.82
    seg["no_speech_prob"] = 0.1
    seg["compression_ratio"] = 1.4
    mock = _mock_whisper(_raw([seg]))
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        result = WhisperTranscriber().transcribe(_audio(10.0))
    out = result["segments"][0]
    assert out["avg_logprob"] == -0.82
    assert out["no_speech_prob"] == 0.1


def test_transcribe_kwargs_override_decode_options() -> None:
    """The self-heal re-decode needs per-call conditioning off."""
    raw = _raw([_seg("moro", [{"word": " moro", "start": 0.0, "end": 0.5}])])
    mock = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t.transcribe(_audio(10.0), condition_on_previous_text=False)

    kwargs = mock.transcribe.call_args.kwargs
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["word_timestamps"] is True  # non-overridden defaults intact


def test_decode_meeting_heals_hallucination_walls() -> None:
    """A wall in the whole-file decode triggers a conditioning-off re-decode
    of the slices under it, and the healed text ships."""
    sr = 16_000
    wall_words = [
        [{"word": " Kiitos.", "start": 10.0 + i, "end": 10.5 + i}] for i in range(8)
    ]
    whole = _raw(
        [
            _seg(
                "alussa ihan oikeaa puhetta tässä on",
                [
                    {"word": " alussa", "start": 0.0, "end": 0.4},
                    {"word": " ihan", "start": 0.5, "end": 0.7},
                    {"word": " oikeaa", "start": 0.8, "end": 1.1},
                    {"word": " puhetta", "start": 1.2, "end": 1.5},
                    {"word": " tässä", "start": 1.6, "end": 1.8},
                    {"word": " on", "start": 1.9, "end": 2.0},
                ],
            )
        ]
        + [_seg("Kiitos.", w) for w in wall_words]
    )
    fixed = _raw(
        [
            _seg(
                "demossa näytettiin ihan oikeita lukuja kaikille",
                [
                    {"word": " demossa", "start": 0.5, "end": 1.0},
                    {"word": " näytettiin", "start": 1.1, "end": 1.6},
                    {"word": " ihan", "start": 1.7, "end": 1.9},
                    {"word": " oikeita", "start": 2.0, "end": 2.4},
                    {"word": " lukuja", "start": 2.5, "end": 2.9},
                    {"word": " kaikille", "start": 3.0, "end": 3.4},
                ],
            )
        ]
    )
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[whole, fixed])
    slices = [
        (0, np.zeros(8 * sr, dtype=np.float32)),
        (9 * sr, np.zeros(11 * sr, dtype=np.float32)),
    ]
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        result = decode_meeting(_audio(20.0), slices)

    assert result is not None
    assert mock.transcribe.call_count == 2
    heal_kwargs = mock.transcribe.call_args_list[1].kwargs
    assert heal_kwargs["condition_on_previous_text"] is False
    assert "Kiitos" not in result["text"]
    assert "demossa" in result["text"]
    # healed words shifted onto the recording timeline (slice offset 9s)
    healed_word = next(w for w in result["words"] if w["word"] == "demossa")
    assert healed_word["start"] == 9.5
