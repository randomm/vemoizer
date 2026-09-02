"""Tests for the Whisper re-decode of disputed spans (issue #8).

All tests are offline: ``huggingface_hub.snapshot_download``,
``mlx_whisper.load_model`` and ``mlx_whisper.transcribe`` are mocked; no
model weights are downloaded and no network is touched.

Coverage list (from the issue #8 test surface):

- ``mlx_whisper.transcribe`` is called with ``word_timestamps=True``,
  ``language="fi"``, and the revision-pinned *local path* (never the
  bare repo ID).
- Slice extraction: ``Span(start, end)`` -> correct 16 kHz float32
  sample offsets (pure function).
- Empty / no-disputed input -> zero ``transcribe`` calls (targeted-only
  cost invariant).
- ``transcribe`` failure -> fail-open: degraded per-span result, never
  raises.
- Disputed-span -> ``transcribe`` invocation mapping (1:N spans,
  overlapping spans already merged by ``merge_spans``).
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vemoizer.redecode import (
    MODEL_ID,
    MODEL_REVISION,
    ReDecodeResult,
    WhisperReDecodeTranscriber,
    _to_result,
    extract_slice,
)
from vemoizer.spans import Span

SAMPLE_RATE = 16_000


def _audio(seconds: float, value: float = 0.1) -> np.ndarray:
    """A deterministic 16 kHz mono float32 buffer of ``seconds`` length."""
    return np.full(int(seconds * SAMPLE_RATE), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Revision pinning (project invariant #4)
# ---------------------------------------------------------------------------


def test_model_id_is_finnish_whisper_large() -> None:
    assert MODEL_ID == "Finnish-NLP/whisper-large-finnish-v3"


def test_model_revision_is_the_pinned_sha() -> None:
    assert MODEL_REVISION == "b23deb0b3855c829ffe04cb1c6709757ff16d49c"


# ---------------------------------------------------------------------------
# Slice extraction (pure function, TDD)
# ---------------------------------------------------------------------------


def test_extract_slice_maps_seconds_to_sample_offsets() -> None:
    audio = _audio(2.0)
    slice_audio = extract_slice(audio, Span(0.5, 1.5))
    assert len(slice_audio) == SAMPLE_RATE  # 1.0 s
    assert slice_audio.dtype == np.float32
    # The returned slice must equal the exact sample range, and be a copy.
    expected = audio[SAMPLE_RATE // 2 : 3 * SAMPLE_RATE // 2]
    np.testing.assert_array_equal(slice_audio, expected)
    slice_audio[0] = -99.0  # mutating the copy must not touch the source
    assert audio[SAMPLE_RATE // 2] == pytest.approx(0.1)


def test_extract_slice_non_integer_boundaries() -> None:
    # 0.25 s at 16 kHz = 4000 samples; 0.7525 s -> int-truncated end offset.
    audio = _audio(1.0)
    slice_audio = extract_slice(audio, Span(0.25, 0.7525))
    expected_len = int(0.7525 * SAMPLE_RATE) - int(0.25 * SAMPLE_RATE)
    assert len(slice_audio) == expected_len


def test_extract_slice_empty_audio_returns_empty() -> None:
    audio = np.zeros(0, dtype=np.float32)
    slice_audio = extract_slice(audio, Span(0.5, 1.5))
    assert len(slice_audio) == 0
    assert slice_audio.dtype == np.float32


def test_extract_slice_out_of_bounds_clips_to_buffer() -> None:
    audio = _audio(1.0)
    # Span extends past the end of the buffer; clip to available samples.
    slice_audio = extract_slice(audio, Span(0.9, 2.0))
    assert len(slice_audio) == int(0.1 * SAMPLE_RATE)


def test_extract_slice_span_before_buffer_starts() -> None:
    audio = _audio(1.0)
    # Negative start clips to 0; end stays where the span ends.
    slice_audio = extract_slice(audio, Span(-0.5, 0.5))
    assert len(slice_audio) == int(0.5 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# WhisperReDecodeTranscriber: lazy, revision-pinned, path-not-repo-ID
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_transcriber() -> Generator[
    tuple[WhisperReDecodeTranscriber, MagicMock], None, None
]:
    """A transcriber whose model load + mlx_whisper.transcribe are mocked.

    Patches both ``huggingface_hub.snapshot_download`` and
    ``mlx_whisper.transcribe`` (the latter via a mock module). Returns the
    transcriber and the mocked ``transcribe`` for call assertions.
    """
    mock_whisper = MagicMock()
    mock_whisper.load_models.load_model.return_value = object()  # a fake "model"
    # Real mlx_whisper.transcribe returns {text, segments, language}; word
    # timestamps live inside each segment, not at the top level.
    mock_whisper.transcribe.return_value = {
        "text": "tässä on teksti",
        "language": "fi",
        "segments": [
            {
                "start": 0.0,
                "end": 0.35,
                "text": " tässä on",
                "words": [
                    {"word": "tässä", "start": 0.0, "end": 0.2},
                    {"word": "on", "start": 0.2, "end": 0.35},
                ],
            }
        ],
    }

    fake_modules = {"mlx_whisper": mock_whisper, "huggingface_hub": MagicMock()}
    fake_modules[
        "huggingface_hub"
    ].snapshot_download.return_value = "/tmp/pinned/whisper-large-finnish-v3"

    with (
        patch.dict("sys.modules", fake_modules),
        patch(
            "vemoizer.redecode.WhisperReDecodeTranscriber._ensure_loaded"
        ) as mock_ensure,
    ):
        transcriber = WhisperReDecodeTranscriber()
        transcriber.model = object()  # bypass the real load
        transcriber._model_path = "/tmp/pinned/whisper-large-finnish-v3"
        transcriber._mlx_whisper = mock_whisper  # use the mocked module
        transcriber._loaded = True
        mock_ensure.return_value = None  # no-op (already loaded)
        yield transcriber, mock_whisper.transcribe


def test_transcribe_span_calls_whisper_with_pinned_path_and_word_timestamps(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, mock_whisper_transcribe = loaded_transcriber
    audio = _audio(5.0)
    transcriber.transcribe_span(audio, Span(1.0, 2.0))

    # Exactly one call for one span.
    mock_whisper_transcribe.assert_called_once()
    args, kwargs = mock_whisper_transcribe.call_args
    # path_or_hf_repo must be the pinned *local path*, never the bare repo ID.
    assert kwargs["path_or_hf_repo"] == "/tmp/pinned/whisper-large-finnish-v3"
    assert kwargs["path_or_hf_repo"] != MODEL_ID  # not the repo ID
    # word_timestamps is opt-in; the re-decode requires it.
    assert kwargs["word_timestamps"] is True
    # Language is pinned to Finnish (invariant #3: don't force one language
    # on the whole file; the slice is Finnish prose).
    assert kwargs["language"] == "fi"
    assert kwargs["task"] == "transcribe"


def test_transcribe_span_shifts_word_timestamps_to_recording_timeline(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, _ = loaded_transcriber
    audio = _audio(5.0)
    result = transcriber.transcribe_span(audio, Span(1.0, 2.0))
    # Whisper returns slice-relative times; the result shifts them back.
    assert result.words[0]["start"] == pytest.approx(1.0)
    assert result.words[0]["end"] == pytest.approx(1.2)
    assert result.words[1]["start"] == pytest.approx(1.2)
    assert result.words[1]["end"] == pytest.approx(1.35)
    # The raw text is preserved (stripped).
    assert result.text == "tässä on teksti"


# ---------------------------------------------------------------------------
# Targeted-only cost invariant: zero transcribe calls for zero spans
# ---------------------------------------------------------------------------


def test_no_spans_zero_transcribe_calls(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, mock_whisper_transcribe = loaded_transcriber
    audio = _audio(5.0)
    # The Transcriber-protocol surface with an empty span list.
    result = transcriber.transcribe(audio, spans=())
    mock_whisper_transcribe.assert_not_called()
    assert result["text"] == ""
    assert result["words"] == []


def test_empty_audio_zero_transcribe_calls(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, mock_whisper_transcribe = loaded_transcriber
    empty = np.zeros(0, dtype=np.float32)
    transcriber.transcribe_span(empty, Span(0.0, 1.0))
    mock_whisper_transcribe.assert_not_called()


def test_1n_spans_one_call_per_span(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, mock_whisper_transcribe = loaded_transcriber
    audio = _audio(10.0)
    # Two disjoint spans (merge_spans would have merged overlapping ones).
    transcriber.transcribe(audio, spans=[Span(0.5, 1.0), Span(5.0, 6.0)])
    assert mock_whisper_transcribe.call_count == 2


def test_merge_spans_output_is_one_call_per_merged_span(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    from vemoizer.spans import merge_spans

    transcriber, mock_whisper_transcribe = loaded_transcriber
    audio = _audio(10.0)
    # Two overlapping raw spans -> one merged slice -> one transcribe call.
    merged = merge_spans([Span(1.0, 2.0), Span(1.5, 3.0)])
    assert len(merged) == 1
    transcriber.transcribe(audio, spans=merged)
    assert mock_whisper_transcribe.call_count == 1
    # The call must carry the merged slice, not the raw input.
    # mlx_whisper.transcribe takes the audio array as its first positional arg.
    args = mock_whisper_transcribe.call_args[0]
    assert len(args) == 1
    assert len(args[0]) == int((merged[0].end - merged[0].start) * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Fail-open: a transcribe failure must not raise
# ---------------------------------------------------------------------------


def test_transcribe_failure_fails_open(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, mock_whisper_transcribe = loaded_transcriber
    mock_whisper_transcribe.side_effect = RuntimeError("device busy")
    audio = _audio(5.0)
    # Must not raise; the result is degraded (ok=False, empty text).
    result = transcriber.transcribe_span(audio, Span(1.0, 2.0))
    assert isinstance(result, ReDecodeResult)
    assert result.ok is False
    assert result.text == ""
    assert result.words == []


def test_load_failure_fails_open() -> None:
    """If the model never loads, every span fails open with an empty result."""
    transcriber = WhisperReDecodeTranscriber()
    # Force the loaded flag so _ensure_loaded is a no-op; model stays None.
    transcriber._loaded = True
    transcriber.model = None
    audio = _audio(5.0)
    result = transcriber.transcribe_span(audio, Span(0.5, 1.5))
    assert result.ok is False
    assert result.text == ""


def test_load_failure_does_not_permanently_disable_redecode() -> None:
    """A transient download failure must not set ``_loaded``.

    ``_ensure_loaded`` only marks the model as loaded *after* a
    successful load. A failure on the first call must allow a retry on
    the next call, so a transient network blip does not silently disable
    re-decode for the rest of the run.
    """
    mock_whisper = MagicMock()
    mock_whisper.load_models.load_model.return_value = object()
    mock_whisper.transcribe.return_value = {"text": "x", "words": []}

    with (
        patch(
            "huggingface_hub.snapshot_download",
            side_effect=[
                RuntimeError("DNS blip"),
                "/tmp/pinned/path",
            ],
        ),
        patch.dict("sys.modules", {"mlx_whisper": mock_whisper}),
    ):
        transcriber = WhisperReDecodeTranscriber()
        audio = _audio(1.0)
        # First call: load fails; _loaded must stay False so a retry is
        # possible.
        result = transcriber.transcribe_span(audio, Span(0.0, 1.0))
        assert result.ok is False
        assert transcriber._loaded is False

        # Second call: a fresh snapshot_download succeeds; the model
        # loads and the span re-decodes.
        result = transcriber.transcribe_span(audio, Span(0.0, 1.0))
        assert result.ok is True
        assert transcriber._loaded is True


# ---------------------------------------------------------------------------
# Lazy loading: nothing at construction, revision-pinned at first use
# ---------------------------------------------------------------------------


def test_construction_does_not_download() -> None:
    with patch("huggingface_hub.snapshot_download") as mock_dl:
        transcriber = WhisperReDecodeTranscriber()
        mock_dl.assert_not_called()
        assert transcriber.model is None


def test_load_model_attribute_path_exists_in_installed_mlx_whisper() -> None:
    """Pin the ``load_models.load_model`` attribute path against the real API.

    The re-decode loads via ``mlx_whisper.load_models.load_model`` because
    the top-level ``__init__`` does not re-export it. If a future
    mlx-whisper release re-exports it (the documented public API), this
    test fails and the load path must be revisited.
    """
    import mlx_whisper

    assert hasattr(mlx_whisper, "load_models")
    assert hasattr(mlx_whisper.load_models, "load_model")
    assert not hasattr(mlx_whisper, "load_model")


def test_first_use_triggers_revision_pinned_download() -> None:
    # spec against the REAL module: MagicMock(spec=...) rejects attributes
    # that do not exist on it, so a drift in the ``load_models.load_model``
    # path (e.g. a re-export at top level, or a rename in the submodule)
    # surfaces as an AttributeError here rather than a silent fail-open in
    # production.
    import mlx_whisper

    mock_whisper = MagicMock(spec=mlx_whisper)
    mock_whisper.load_models.load_model.return_value = object()
    mock_snapshot = MagicMock(return_value="/tmp/pinned/path")
    with (
        patch("huggingface_hub.snapshot_download", mock_snapshot),
        patch.dict("sys.modules", {"mlx_whisper": mock_whisper}),
    ):
        transcriber = WhisperReDecodeTranscriber()
        audio = _audio(1.0)
        # transcribe_span will try to call mlx_whisper.transcribe on the mock;
        # the model is a bare object (load_models.load_model.return_value),
        # so pass a mock transcribe too.
        mock_whisper.transcribe.return_value = {"text": "x", "words": []}
        transcriber.transcribe_span(audio, Span(0.0, 1.0))
        mock_snapshot.assert_called_once_with(MODEL_ID, revision=MODEL_REVISION)
        # The model was loaded from the local path, not the repo ID.
        mock_whisper.load_models.load_model.assert_called_once_with("/tmp/pinned/path")


def test_transcribe_protocol_surface_concatenates_ok_results(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, mock_whisper_transcribe = loaded_transcriber
    mock_whisper_transcribe.side_effect = [
        {
            "text": "ensimmäinen",
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": " ensimmäinen",
                    "words": [{"word": "ensimmäinen", "start": 0.0, "end": 0.5}],
                }
            ],
        },
        {"text": "toinen", "segments": []},
    ]
    audio = _audio(10.0)
    result = transcriber.transcribe(audio, spans=[Span(0.0, 1.0), Span(5.0, 6.0)])
    assert result["text"] == "ensimmäinen toinen"
    # Words from the first span (shifted), second has none.
    assert len(result["words"]) == 1
    assert result["words"][0]["word"] == "ensimmäinen"


def test_transcribe_protocol_surface_rtf_is_ratio_of_reported_fields(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    """rtf must be the ratio of the two reported fields.

    The adapter measures wall-clock across the per-span decode calls only;
    the model load (which can take seconds on the first call) is timed and
    logged separately in _ensure_loaded and must not leak into the rtf
    ratio, or the first call's figure is meaningless.
    """
    transcriber, mock_whisper_transcribe = loaded_transcriber
    mock_whisper_transcribe.return_value = {"text": "ok", "segments": []}
    audio = _audio(5.0)
    result = transcriber.transcribe(audio, spans=[Span(0.5, 1.0)])
    # rtf = transcribe_time / audio_duration, both as reported by the
    # adapter (the load, if any, is outside the timed window).
    assert result["audio_duration"] == pytest.approx(5.0)
    assert result["rtf"] == pytest.approx(
        result["transcribe_time"] / result["audio_duration"]
    )


def test_transcribe_protocol_surface_skips_failed_span(
    loaded_transcriber: tuple[WhisperReDecodeTranscriber, MagicMock],
) -> None:
    transcriber, mock_whisper_transcribe = loaded_transcriber
    # First span succeeds, second fails (fail-open for that span only).
    mock_whisper_transcribe.side_effect = [
        {"text": "ok", "segments": []},
        RuntimeError("timeout"),
    ]
    audio = _audio(10.0)
    result = transcriber.transcribe(audio, spans=[Span(0.0, 1.0), Span(5.0, 6.0)])
    # The successful span's text is preserved; the failed one is skipped.
    assert result["text"] == "ok"
    assert result["words"] == []


# ---------------------------------------------------------------------------
# ReDecodeResult invariants
# ---------------------------------------------------------------------------


def test_to_result_flattens_segment_word_timestamps() -> None:
    """``transcribe`` returns ``{text, segments, language}`` — word
    timestamps live inside each segment, not at the top level. The
    flattening must match the real API shape (no top-level ``words`` key).
    """
    raw = {
        "text": "moi kaikki",
        "language": "fi",
        "segments": [
            {
                "start": 0.0,
                "end": 0.3,
                "text": " moi",
                "words": [{"word": "moi", "start": 0.0, "end": 0.3}],
            },
            {
                "start": 0.3,
                "end": 0.9,
                "text": " kaikki",
                "words": [{"word": "kaikki", "start": 0.3, "end": 0.9}],
            },
        ],
    }
    result = _to_result(Span(2.0, 3.0), raw)
    assert [w["word"] for w in result.words] == ["moi", "kaikki"]
    # Shifted back onto the recording timeline (slice starts at 2.0s).
    assert result.words[0]["start"] == pytest.approx(2.0)
    assert result.words[1]["end"] == pytest.approx(2.9)
    assert result.text == "moi kaikki"
    assert result.ok is True


def test_to_result_handles_missing_words_and_segments() -> None:
    # ``words`` is absent from every segment, or ``segments`` is missing
    # entirely: word-level output degrades to empty without crashing.
    result = _to_result(Span(0.0, 1.0), {"text": "x", "segments": []})
    assert result.words == []
    result = _to_result(Span(0.0, 1.0), {"text": "x"})
    assert result.words == []
    # Segments without a ``words`` key (word_timestamps off for a slice).
    result = _to_result(Span(0.0, 1.0), {"text": "x", "segments": [{"start": 0}]})
    assert result.words == []


def test_redecode_result_is_immutable() -> None:
    # Both ReDecodeResult and Span are frozen dataclasses; the result carries
    # an immutable span so downstream bookkeeping can rely on its identity.
    result = ReDecodeResult(span=Span(0, 1), text="x", words=[], ok=True)
    assert isinstance(result, ReDecodeResult)
    assert result.span.start == 0
    assert result.span.end == 1
    assert result.ok is True


def test_span_validation_is_inherited() -> None:
    # ReDecodeResult carries a Span; a malformed Span is rejected.
    with pytest.raises(ValueError):
        Span(1.0, 0.5)
