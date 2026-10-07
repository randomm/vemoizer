"""WhisperTranscriber — turbo decode A for the meeting profile (issue #71, #76).

mlx_whisper is mocked throughout: no downloads, no GPU. The transcriber
decodes the recording in 30 s windows (each its own transcribe() call so
the glossary initial_prompt re-seeds every window — issue #76) and surfaces
words/segments/language; per-VAD-slice records for dispute detection are
derived from the word timestamps.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vemoizer.echo_filter import echo_vocabulary, filter_echo_segments
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


def _kw(call, name: str) -> object:
    """Read one keyword argument from a mocked call.

    MagicMock's ``call_args_list`` iteration yields objects whose
    ``kwargs`` indexing is not type-checkable, so the index access is
    centralized here (issue #128) instead of suppressing it at every
    per-window assertion.
    """
    return call.kwargs[name]


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
        assert _kw(call, "language") is None


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
        assert _kw(call, "language") == "fi"


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
        assert _kw(call, "language") is None
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
        assert _kw(call, "language") is None


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
        assert _kw(call, "language") == "fi"


# -- verbose kwarg pinning (issue #105, lens MEDIUM) ------------------------


def test_verbose_false_always_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verbose=False is ALWAYS passed to mlx_whisper.transcribe (issue #147),
    regardless of display state: it suppresses the per-window
    "Detected language: X" print in all cases. With an active display the
    shim intercepts the tqdm bar; with no display or a disabled display,
    tqdm auto-suppresses in non-TTY contexts."""
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

    from vemoizer.progress import ProgressDisplay

    # Case 1: active display → verbose=False
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
    assert mock_on.transcribe.call_args.kwargs.get("verbose") is False

    # Case 2: display=None → verbose=False
    mock_off = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock_off}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock_off
        t.transcribe(_audio(60.0))
    assert mock_off.transcribe.call_args.kwargs.get("verbose") is False

    # Case 3: disabled display → verbose=False
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
    assert mock_disabled.transcribe.call_args.kwargs.get("verbose") is False


def test_heal_redecode_never_gets_verbose_true() -> None:
    """The self-heal re-decode (condition_on_previous_text=False) must never
    receive verbose=True. verbose=False is always passed (issue #147) —
    it suppresses the per-window print and is safe for heal re-decodes."""
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
        t.transcribe(_audio(10.0), condition_on_previous_text=False)

    kwargs = mock.transcribe.call_args.kwargs
    assert kwargs.get("verbose") is not True, (
        f"heal re-decode must not get verbose=True, got {kwargs.get('verbose')}"
    )
    assert kwargs.get("verbose") is False, (
        f"Expected verbose=False on heal path, got {kwargs.get('verbose')}"
    )


# -- post-decode echo filter (issue #109) ----------------------------------
#
# Whisper can continue the glossary initial_prompt instead of transcribing
# ("Sanasto, Pia, NG-TOPI, ..." in English meetings). The filter drops those
# echo segments but must keep any real sentence that contains one or more
# glossary terms. Fail-open: a filter error keeps the unfiltered transcript.


def _echo_vocab(prompt):
    return echo_vocabulary(prompt)


def test_echo_vocabulary_none_without_prompt() -> None:
    """No glossary configured => nothing to echo => filter is a no-op."""
    assert _echo_vocab(None) is None
    assert _echo_vocab("   ") is None


def test_echo_vocabulary_includes_terms_and_label() -> None:
    """Terms from the configured prompt (case preserved) + the former label."""
    vocab = _echo_vocab("Pia, NG-TOPI, IBC.")
    assert vocab is not None
    assert "Sanasto" in vocab  # former label, always present
    assert "Pia" in vocab
    assert "NG-TOPI" in vocab
    assert "IBC" in vocab


def test_filter_drops_prompt_label_echo() -> None:
    """A fake decode echoing the former label plus terms is filtered."""
    seg = {
        "text": "Sanasto, Pia, NG-TOPI, IBC.",
        "start": 0.0,
        "end": 1.5,
        "words": [
            {"word": " Sanasto", "start": 0.0, "end": 0.4},
            {"word": " Pia", "start": 0.5, "end": 0.9},
        ],
    }
    vocab = _echo_vocab("Pia, NG-TOPI, IBC.")
    segments, words = filter_echo_segments([seg], 0.0, vocab)
    assert segments == []
    assert words == []


def test_echo_vocabulary_normalizes_hyphenated_terms() -> None:
    """A hyphenated term is kept as-is (case preserved, hyphens intact);
    :func:`_is_echo` matches the term directly when the hyphen is present
    in the text, and falls back to the eval harness's proportional form
    when it isn't (issue #109 review)."""
    vocab = _echo_vocab("NG-TOPI, IBC.")
    assert vocab is not None
    assert "NG-TOPI" in vocab
    assert "IBC" in vocab
    # _is_echo matches the term directly when the hyphen is present:
    from vemoizer.echo_filter import _is_echo, is_echo, vocabulary_set

    vocab_set = vocabulary_set(vocab)
    assert is_echo("NG-TOPI, IBC", vocab) is True
    # When the hyphen is transcribed as a space, the strict form doesn't
    # match (the eval harness's proportional form handles that case).
    assert _is_echo("NG TOPI, IBC", vocab_set) is False


def test_filter_drops_bare_term_run() -> None:
    """A bare 'term, term, term' run (no label) is also an echo."""
    seg = {
        "text": "NG-TOPI, IBC, DCS.",
        "start": 2.0,
        "end": 3.0,
        "words": [{"word": " NG-TOPI", "start": 2.0, "end": 2.5}],
    }
    vocab = _echo_vocab("NG-TOPI, IBC, DCS.")
    segments, words = filter_echo_segments([seg], 0.0, vocab)
    assert segments == []
    assert words == []


def test_filter_keeps_real_sentence_with_one_term() -> None:
    """A real sentence containing one glossary term is kept unchanged."""
    seg = {
        "text": "We need to own the solution for IBC.",
        "start": 0.0,
        "end": 1.2,
        "words": [
            {"word": " We", "start": 0.0, "end": 0.2},
            {"word": " IBC", "start": 0.9, "end": 1.2},
        ],
    }
    vocab = _echo_vocab("IBC, DCS.")
    segments, words = filter_echo_segments([seg], 0.0, vocab)
    assert len(segments) == 1
    assert segments[0]["text"] == "We need to own the solution for IBC."
    assert [w["word"] for w in words] == ["We", "IBC"]


def test_filter_keeps_mixed_segments_drops_echo_only() -> None:
    """Only the echo segment is dropped; a real segment next to it stays."""
    echo = {"text": "Sanasto, DCS.", "start": 0.0, "end": 0.5, "words": []}
    real = {
        "text": "the way we want to go",
        "start": 1.0,
        "end": 1.5,
        "words": [{"word": " the", "start": 1.0, "end": 1.2}],
    }
    vocab = _echo_vocab("DCS.")
    segments, words = filter_echo_segments([echo, real], 30.0, vocab)
    assert len(segments) == 1
    assert segments[0]["text"] == "the way we want to go"
    assert [w["word"] for w in words] == ["the"]


def test_filter_noop_when_no_glossary() -> None:
    """With no glossary (vocab None) nothing is dropped, even term-like text."""
    seg = {"text": "Pia, DCS, IBC.", "start": 0.0, "end": 1.0, "words": []}
    segments, words = filter_echo_segments([seg], 0.0, None)
    assert segments == [seg]
    assert words == []


def test_filter_fail_open_on_error(caplog) -> None:
    """On any filter error the unfiltered segments are returned (fail-open).

    The log line reports the exception type and message only (no traceback)
    so that a malformed payload cannot leak transcript text into the log.
    """
    # A segment whose start is not numeric will raise in the log line's
    # float() conversion -> the except branch returns the unfiltered list.
    seg = {"text": "Sanasto, DCS.", "start": "bogus", "end": 1.0, "words": []}
    vocab = _echo_vocab("DCS.")
    with caplog.at_level(logging.WARNING):
        segments, _ = filter_echo_segments([seg], 0.0, vocab)
    # Fail-open: the segment is not lost.
    assert len(segments) == 1
    assert segments[0]["text"] == "Sanasto, DCS."
    # The warning is logged without a traceback (no exc_info) so that
    # transcript fragments cannot leak into the log via the exception
    # message or stack.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "echo filter error" in warnings[0].getMessage()
    # No traceback in the message (exc_info would set this).
    assert warnings[0].exc_info is None


def test_filter_fail_open_degrades_words_with_warning(caplog) -> None:
    """When the fallback _words_on_timeline also fails, the words degrade
    to empty and a second warning is logged (distinguishable from the
    outer 'returning unfiltered' warning)."""
    # A word with a non-numeric start will raise in _words_on_timeline's
    # float() conversion. The outer catch logs 'returning unfiltered',
    # the inner catch logs 'words extraction failed' with a distinct
    # message so the degradation is observable.
    seg = {
        "text": "Sanasto, DCS.",
        "start": "bogus",  # raises in the outer float() for the drop log
        "end": 1.0,
        "words": [{"word": " Sanasto", "start": "bad", "end": 0.4}],
    }
    vocab = _echo_vocab("DCS.")
    with caplog.at_level(logging.WARNING):
        segments, words = filter_echo_segments([seg], 0.0, vocab)
    # Segments survive (fail-open); words degrade to empty.
    assert len(segments) == 1
    assert segments[0]["text"] == "Sanasto, DCS."
    assert words == []
    # Two distinct warnings: outer + inner degradation.
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 2
    assert "echo filter error" in warnings[0].getMessage()
    assert "words extraction failed" in warnings[1].getMessage()


def test_transcribe_drops_echo_segment_end_to_end() -> None:
    """The full transcribe() drops an echo segment and its words."""
    raw = _raw(
        [
            _seg(
                "Sanasto, NG-TOPI, IBC.",
                [
                    {"word": " Sanasto", "start": 0.0, "end": 0.4},
                    {"word": " NG-TOPI", "start": 0.5, "end": 0.9},
                ],
            ),
            _seg(
                "we want to go that way",
                [
                    {"word": " we", "start": 1.0, "end": 1.2},
                    {"word": " want", "start": 1.3, "end": 1.6},
                ],
            ),
        ]
    )
    mock = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="NG-TOPI, IBC.")
        result = t.transcribe(_audio(10.0))
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "we want to go that way"
    assert "Sanasto" not in " ".join(w["word"] for w in result["words"])


def test_transcribe_no_glossary_keeps_all() -> None:
    """Without a glossary, even a term-list segment is kept (no-op filter)."""
    raw = _raw(
        [
            _seg(
                "Pia, DCS, IBC.",
                [{"word": " Pia", "start": 0.0, "end": 0.4}],
            ),
        ]
    )
    mock = _mock_whisper(raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber()  # no prompt
        result = t.transcribe(_audio(10.0))
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "Pia, DCS, IBC."


def test_prompt_has_no_sanasto_prefix() -> None:
    """The built glossary prompt carries no 'Sanasto' label (issue #109, opt 1).

    Regression guard on the *output* of glossary_prompt (the neutral form
    has no label prefix), not on a module constant.
    """
    from vemoizer.glossary import glossary_prompt

    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [len(t) for t in text.split()]

    prompt = glossary_prompt(["Nordea", "Riihimäki"], _Tok())
    assert prompt is not None
    assert not prompt.startswith("Sanasto")
    assert "Sanasto" not in prompt


# -- per-window language counting + summary (issue #147) --------------------


def _raw_lang(language: str, text: str = "moro") -> dict:
    return {
        "text": text,
        "language": language,
        "segments": [_seg(text, [{"word": " " + text, "start": 0.0, "end": 0.5}])],
    }


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


# -- huggingface_hub progress bar suppression (issue #147) --------------------


def test_pinned_download_suppresses_bars_when_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the snapshot is cached (its ``trees/<revision>.json`` tree cache
    exists), _pinned_download silences the HF bars around snapshot_download
    (issue #147): disable_progress_bars enters before the download and its
    context-manager exit re-enables."""
    from vemoizer.whisper_transcriber import _pinned_download

    revision = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # A full commit-SHA revision never gets a refs/<sha> marker
    # (huggingface_hub writes refs only when revision != commit_hash);
    # the tree cache IS written for a pinned SHA, so that is the marker.
    tree = tmp_path / "hub" / "models--org--name" / "trees" / f"{revision}.json"
    tree.parent.mkdir(parents=True)
    tree.touch()

    calls: list[str] = []

    def _fake_download(repo_id: str, revision: str) -> str:
        calls.append("download")
        return str(tmp_path / "snap")

    with (
        patch("vemoizer.models._model_cache_name", return_value="models--org--name"),
        patch("huggingface_hub.snapshot_download", side_effect=_fake_download),
        patch("huggingface_hub.utils.disable_progress_bars.__enter__") as mock_enter,
        patch("huggingface_hub.utils.disable_progress_bars.__exit__") as mock_exit,
    ):
        # The __enter__ patch must return the cm so `with` gets a truthy object.
        mock_enter.side_effect = lambda: calls.append("disable") or object()
        mock_exit.side_effect = lambda *a: calls.append("enable") or None
        result = _pinned_download("org/name", revision)

    assert result == str(tmp_path / "snap")
    assert "disable" in calls
    assert "enable" in calls
    assert calls.index("disable") < calls.index("download")
    assert calls.index("download") < calls.index("enable")


def test_pinned_download_keeps_bars_when_uncached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the snapshot is NOT cached (no tree cache), _pinned_download does
    NOT call disable_progress_bars (a real multi-GB download must show a
    bar)."""
    from vemoizer.whisper_transcriber import _pinned_download

    revision = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # No trees/<revision>.json marker -> not cached
    with (
        patch("vemoizer.models._model_cache_name", return_value="models--org--name"),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/snap"),
        patch("huggingface_hub.utils.disable_progress_bars.__enter__") as mock_disable,
        patch("huggingface_hub.utils.disable_progress_bars.__exit__") as mock_exit,
    ):
        _pinned_download("org/name", revision)
    mock_disable.assert_not_called()
    mock_exit.assert_not_called()
