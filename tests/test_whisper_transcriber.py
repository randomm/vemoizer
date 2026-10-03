"""WhisperTranscriber — turbo decode A for the meeting profile (issue #71, #76).

mlx_whisper is mocked throughout: no downloads, no GPU. The transcriber
decodes the recording in 30 s windows (each its own transcribe() call so
the glossary initial_prompt re-seeds every window — issue #76) and surfaces
words/segments/language; per-VAD-slice records for dispute detection are
derived from the word timestamps.
"""

from __future__ import annotations

import sys
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


def test_transcribe_decodes_each_window_separately() -> None:
    """Each 30 s window is its own transcribe() call so the glossary
    initial_prompt re-seeds it (issue #76: a single whole-file call would
    let the rolling context slide the glossary past the 223-token
    keep-window)."""
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
        t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
        t.transcribe(_audio(120.0))

    assert mock.transcribe.call_count == 4  # 4 × 30 s windows
    first_kwargs = mock.transcribe.call_args_list[0].kwargs
    last_kwargs = mock.transcribe.call_args_list[-1].kwargs
    assert first_kwargs["path_or_hf_repo"] == "/tmp/turbo"
    assert first_kwargs["word_timestamps"] is True
    assert first_kwargs["temperature"][0] == 0.0  # deterministic-first ladder
    assert first_kwargs["condition_on_previous_text"] is True
    assert first_kwargs["initial_prompt"] == "Sanasto: Flagship."
    assert last_kwargs["initial_prompt"] == "Sanasto: Flagship."  # re-seeds each window
    # The default is per-window language detection (issue #108, option A):
    # language=None on EVERY window, never a file-level pin.
    for call in mock.transcribe.call_args_list:
        assert call.kwargs["language"] is None  # type: ignore[index]


def test_transcribe_language_override_pins_every_window() -> None:
    """A run-level override (issue #108, option B) pins the language on
    every window, not just the first."""
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
        t = WhisperTranscriber(language="fi")
        t.transcribe(_audio(60.0))

    assert mock.transcribe.call_count == 2
    for call in mock.transcribe.call_args_list:
        assert call.kwargs["language"] == "fi"  # type: ignore[index]


def test_window_returning_none_raises_with_window_index() -> None:
    """A None result from mlx_whisper.transcribe (transient GPU / MLX
    memory fault) must not die mid-loop on an AttributeError; the failing
    window index must be named."""
    raw = _raw(
        [
            _seg(
                "moro",
                [{"word": " moro", "start": 0.0, "end": 0.5}],
            )
        ]
    )
    mock = _mock_whisper(None)
    mock.transcribe.side_effect = [raw, None]
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        with pytest.raises(RuntimeError, match="window 1"):
            t.transcribe(_audio(60.0))


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
    # Rolling context ON within each window; the glossary itself re-seeds at
    # every window boundary (each window is its own transcribe() call), so
    # the prompt does not depend on the rolling context carrying it.
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
    # Option A: decode_meeting's default leaves per-window detection on.
    for call in mock.transcribe.call_args_list:
        assert call.kwargs["language"] is None  # type: ignore[index]
    assert "Kiitos" not in result["text"]
    assert "demossa" in result["text"]
    # healed words shifted onto the recording timeline (slice offset 9s)
    healed_word = next(w for w in result["words"] if w["word"] == "demossa")
    assert healed_word["start"] == 9.5


def test_decode_meeting_falls_back_to_prompt_free_redecode() -> None:
    """When the prompted re-decode still loops (the prompt itself being
    echoed), the fallback re-decodes the slice without the glossary."""
    sr = 16_000
    wall = [
        _seg("Janni.", [{"word": " Janni.", "start": 1.0 + i, "end": 1.5 + i}])
        for i in range(8)
    ]
    still_looping = _raw(
        [
            _seg("Janni.", [{"word": " Janni.", "start": 0.5 + i, "end": 0.9 + i}])
            for i in range(6)
        ]
    )
    clean = _raw(
        [
            _seg(
                "we talked about the data platform today",
                [{"word": " we", "start": 0.5, "end": 0.7}],
            )
        ]
    )
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[_raw(wall), still_looping, clean])
    slices = [(0, np.zeros(10 * sr, dtype=np.float32))]
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        result = decode_meeting(_audio(10.0), slices, initial_prompt="Sanasto: Janni.")

    assert result is not None
    assert mock.transcribe.call_count == 3
    primary, fallback = (c.kwargs for c in mock.transcribe.call_args_list[1:])
    assert primary["initial_prompt"] == "Sanasto: Janni."
    assert fallback["initial_prompt"] is None
    assert fallback["condition_on_previous_text"] is False
    for call in mock.transcribe.call_args_list:
        assert call.kwargs["language"] is None  # type: ignore[index]


def test_decode_meeting_language_kwarg_pins_every_window() -> None:
    """Option B: decode_meeting(language="fi") pins the language on the
    primary decode AND the self-heal re-decode windows — a pin must not
    silently drop out of the heal path."""
    raw = _raw([_seg("moro", [{"word": " moro", "start": 0.0, "end": 0.5}])])
    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[raw, raw])
    slices = [(0, np.zeros(8 * 16_000, dtype=np.float32))]
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        result = decode_meeting(_audio(20.0), slices, language="fi")

    assert result is not None
    assert mock.transcribe.call_count >= 1
    for call in mock.transcribe.call_args_list:
        assert call.kwargs["language"] == "fi"  # type: ignore[index]


# -- verbose kwarg pinning (issue #105, lens MEDIUM) ------------------------


def test_verbose_false_only_when_display_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an active (non-disabled) display, every window call receives
    verbose=False (which enables the real tqdm bar in mlx-whisper 0.4.3 —
    the shim intercepts it). With display=None or a disabled display,
    NO verbose kwarg is passed (library default: no bar, no print)."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
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

    # Case 1: active display → verbose=False passed
    from vemoizer.progress import ProgressDisplay

    display_on = ProgressDisplay(verbose=True)
    display_on.start()
    mock_on = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock_on}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock_on
        t.transcribe(_audio(60.0), display=display_on)
    display_on.close()
    kwargs_on = mock_on.transcribe.call_args.kwargs
    assert kwargs_on.get("verbose") is False, (
        f"Expected verbose=False with active display, got {kwargs_on.get('verbose')}"
    )

    # Case 2: display=None → no verbose kwarg
    mock_off = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock_off}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock_off
        t.transcribe(_audio(60.0))
    kwargs_off = mock_off.transcribe.call_args.kwargs
    assert "verbose" not in kwargs_off, (
        f"Expected no verbose kwarg with display=None, got {kwargs_off.get('verbose')}"
    )

    # Case 3: disabled display → no verbose kwarg
    display_disabled = ProgressDisplay(verbose=False)  # disable=True
    mock_disabled = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock_disabled}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock_disabled
        t.transcribe(_audio(60.0), display=display_disabled)
    kwargs_disabled = mock_disabled.transcribe.call_args.kwargs
    assert "verbose" not in kwargs_disabled, (
        f"Expected no verbose kwarg with disabled display, "
        f"got {kwargs_disabled.get('verbose')}"
    )


def test_heal_redecode_never_gets_verbose_true() -> None:
    """The self-heal re-decode (condition_on_previous_text=False) must never
    receive verbose=True. The heal path calls transcribe with display=None
    (the heal lambdas do not pass a display), so no verbose kwarg is set."""
    raw = _raw(
        [
            _seg(
                "moro",
                [{"word": " moro", "start": 0.0, "end": 0.5}],
            )
        ]
    )
    mock = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        # Simulate the heal re-decode call
        t.transcribe(_audio(10.0), condition_on_previous_text=False)

    kwargs = mock.transcribe.call_args.kwargs
    assert kwargs.get("verbose") is not True, (
        f"heal re-decode must not get verbose=True, got {kwargs.get('verbose')}"
    )
    # The heal path does not pass a display, so no verbose kwarg
    assert "verbose" not in kwargs, (
        f"Expected no verbose kwarg on heal path, got {kwargs.get('verbose')}"
    )
