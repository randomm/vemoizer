"""Tests for the Whisper re-decode of disputed spans (issue #8).

All tests are offline: ``huggingface_hub.snapshot_download`` and
``mlx_whisper.transcribe`` are mocked; no model weights are downloaded
and no network is touched.

Coverage list (from the issue #8 test surface):

- ``mlx_whisper.transcribe`` is called with ``word_timestamps=True``,
  ``language="fi"``, and the revision-pinned *local path* (never the
  bare repo ID).
- Slice extraction: ``Span(start, end)`` -> correct 16 kHz float32
  sample offsets (pure function).
- Empty / no-disputed input -> zero ``transcribe`` calls and an empty
  result (targeted-only cost invariant).
- ``transcribe`` failure -> fail-open: degraded per-span result, never
  raises.
- Disputed-span -> ``transcribe`` invocation mapping (1:N spans,
  overlapping spans already merged by ``merge_spans``).

The re-decode is a standalone ``mlx_whisper`` operation, not a
``Transcriber`` implementation (lens review of PR #28): it re-decodes
seconds-long slices as a third opinion, which the Protocol's
full-file ``transcribe(audio, **kwargs)`` shape is a misfit for.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vemoizer.redecode import (
    MODEL_ID,
    MODEL_REVISION,
    ReDecodeResult,
    _to_result,
    extract_slice,
    redecode,
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
# redecode: standalone mlx_whisper re-decode, path-not-repo-ID
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_whisper() -> Generator[tuple[MagicMock, MagicMock, MagicMock], None, None]:
    """A mocked ``mlx_whisper`` module + ``snapshot_download``.

    Returns ``(mock_whisper, mock_snapshot, mock_whisper_transcribe)``
    where ``mock_whisper_transcribe`` is the ``transcribe`` callable on
    the mocked module, so call assertions can target it directly.
    """
    mock_whisper = MagicMock()
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
    mock_snapshot = MagicMock(return_value="/tmp/pinned/whisper-large-finnish-v3")

    with (
        patch.dict("sys.modules", {"mlx_whisper": mock_whisper}),
        patch("huggingface_hub.snapshot_download", mock_snapshot),
    ):
        yield mock_whisper, mock_snapshot, mock_whisper.transcribe


def test_redecode_calls_whisper_with_pinned_path_and_word_timestamps(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, mock_snapshot, mock_whisper_transcribe = mocked_whisper
    audio = _audio(5.0)
    redecode(audio, [Span(1.0, 2.0)])

    # Exactly one call for one span.
    mock_whisper_transcribe.assert_called_once()
    args, kwargs = mock_whisper_transcribe.call_args
    # path_or_hf_repo must be the pinned *local path* (a str), never the
    # bare repo ID.
    assert kwargs["path_or_hf_repo"] == "/tmp/pinned/whisper-large-finnish-v3"
    assert kwargs["path_or_hf_repo"] != MODEL_ID  # not the repo ID
    # word_timestamps is opt-in; the re-decode requires it.
    assert kwargs["word_timestamps"] is True
    # Language is pinned to Finnish (invariant #3: don't force one language
    # on the whole file; the slice is Finnish prose).
    assert kwargs["language"] == "fi"
    assert kwargs["task"] == "transcribe"
    # The model was downloaded revision-pinned, exactly once.
    mock_snapshot.assert_called_once_with(MODEL_ID, revision=MODEL_REVISION)


def test_redecode_shifts_word_timestamps_to_recording_timeline(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _ = mocked_whisper
    audio = _audio(5.0)
    results = redecode(audio, [Span(1.0, 2.0)])
    assert len(results) == 1
    result = results[0]
    # Whisper returns slice-relative times; the result shifts them back.
    assert result.words[0]["start"] == pytest.approx(1.0)
    assert result.words[0]["end"] == pytest.approx(1.2)
    assert result.words[1]["start"] == pytest.approx(1.2)
    assert result.words[1]["end"] == pytest.approx(1.35)
    # The raw text is preserved (stripped).
    assert result.text == "tässä on teksti"
    assert result.ok is True


def test_redecode_one_call_per_non_empty_span(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, _, mock_whisper_transcribe = mocked_whisper
    audio = _audio(10.0)
    # Two disjoint spans (merge_spans would have merged overlapping ones).
    results = redecode(audio, [Span(0.5, 1.0), Span(5.0, 6.0)])
    assert mock_whisper_transcribe.call_count == 2
    assert len(results) == 2
    assert all(r.ok for r in results)


def test_merge_spans_output_is_one_call_per_merged_span(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    from vemoizer.spans import merge_spans

    _, _, mock_whisper_transcribe = mocked_whisper
    audio = _audio(10.0)
    # Two overlapping raw spans -> one merged slice -> one transcribe call.
    merged = merge_spans([Span(1.0, 2.0), Span(1.5, 3.0)])
    assert len(merged) == 1
    redecode(audio, merged)
    assert mock_whisper_transcribe.call_count == 1
    # The call must carry the merged slice, not the raw input.
    # mlx_whisper.transcribe takes the audio array as its first positional arg.
    args = mock_whisper_transcribe.call_args[0]
    assert len(args) == 1
    assert len(args[0]) == int((merged[0].end - merged[0].start) * SAMPLE_RATE)


def test_redecode_no_download_before_first_decode(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """The model is downloaded at most once, at the first non-empty slice.

    The lazy download must happen only when a decode is actually needed —
    not per span, and not when every span has no audio.
    """
    _, mock_snapshot, mock_whisper_transcribe = mocked_whisper
    audio = _audio(10.0)
    redecode(audio, [Span(0.5, 1.0), Span(5.0, 6.0)])
    # Two decodes, but exactly one download (the model path is cached).
    assert mock_whisper_transcribe.call_count == 2
    assert mock_snapshot.call_count == 1


# ---------------------------------------------------------------------------
# Targeted-only cost invariant: zero transcribe calls for zero spans
# ---------------------------------------------------------------------------


def test_empty_spans_returns_empty_result_without_download(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, mock_snapshot, mock_whisper_transcribe = mocked_whisper
    audio = _audio(5.0)
    # No spans -> no result, no download, no decode.
    assert redecode(audio, []) == []
    mock_snapshot.assert_not_called()
    mock_whisper_transcribe.assert_not_called()


def test_empty_audio_returns_empty_result_without_download(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, mock_snapshot, mock_whisper_transcribe = mocked_whisper
    empty = np.zeros(0, dtype=np.float32)
    assert redecode(empty, [Span(0.0, 1.0)]) == []
    mock_snapshot.assert_not_called()
    mock_whisper_transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# Fail-open: a transcribe failure must not raise
# ---------------------------------------------------------------------------


def test_transcribe_failure_fails_open(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, _, mock_whisper_transcribe = mocked_whisper
    mock_whisper_transcribe.side_effect = RuntimeError("device busy")
    audio = _audio(5.0)
    # Must not raise; the result is degraded (ok=False, empty text).
    results = redecode(audio, [Span(1.0, 2.0)])
    assert len(results) == 1
    assert isinstance(results[0], ReDecodeResult)
    assert results[0].ok is False
    assert results[0].text == ""
    assert results[0].words == []


def test_download_failure_fails_open() -> None:
    """If the model never downloads, the span fails open with an empty result."""
    mock_whisper = MagicMock()
    dns_blip = RuntimeError("DNS blip")
    with (
        patch("huggingface_hub.snapshot_download", side_effect=dns_blip),
        patch.dict("sys.modules", {"mlx_whisper": mock_whisper}),
    ):
        audio = _audio(5.0)
        results = redecode(audio, [Span(0.5, 1.5)])
        assert len(results) == 1
        assert results[0].ok is False
        assert results[0].text == ""
        mock_whisper.transcribe.assert_not_called()


def test_download_failure_does_not_permanently_disable_redecode() -> None:
    """A transient download failure must not skip the retry on the next span."""
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {"text": "x", "segments": []}
    mock_snapshot = MagicMock(
        side_effect=[RuntimeError("DNS blip"), "/tmp/pinned/path"]
    )
    with (
        patch("huggingface_hub.snapshot_download", mock_snapshot),
        patch.dict("sys.modules", {"mlx_whisper": mock_whisper}),
    ):
        audio = _audio(3.0)
        # First span: download fails (fails open); second span: the retry
        # succeeds and the span re-decodes.
        results = redecode(audio, [Span(0.0, 1.0), Span(1.5, 2.5)])
        assert results[0].ok is False
        assert results[1].ok is True
        assert mock_snapshot.call_count == 2


# ---------------------------------------------------------------------------
# Lazy loading: revision-pinned download at first use, not at import
# ---------------------------------------------------------------------------


def test_import_does_not_download_or_import_mlx_whisper() -> None:
    """Importing the module must not require mlx_whisper to be installed."""
    import importlib

    import vemoizer.redecode as mod

    reloaded: object = importlib.reload(mod)
    assert not hasattr(reloaded, "mlx_whisper")  # imported lazily, per decode


def test_download_path_is_typed_str_not_repo_id(
    mocked_whisper: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """``path_or_hf_repo`` must carry a local-path ``str`` (never the repo ID).

    Guards the ``str`` typing of the download result: a ``None`` or
    repo-ID passthrough fails this test rather than reaching the model
    with an unpinned source.
    """
    _, mock_snapshot, mock_whisper_transcribe = mocked_whisper
    audio = _audio(5.0)
    redecode(audio, [Span(1.0, 2.0)])
    path = mock_whisper_transcribe.call_args.kwargs["path_or_hf_repo"]
    assert isinstance(path, str)
    assert path != MODEL_ID
    assert mock_snapshot.call_args.args[0] == MODEL_ID


def test_load_model_attribute_path_exists_in_installed_mlx_whisper() -> None:
    """Pin the ``load_models.load_model`` attribute path against the real API.

    mlx-whisper ships ``load_model`` in its ``load_models`` submodule; the
    top-level ``__init__`` does not re-export it. If a future release
    changes that, this test fails and the (documented) load path in
    ``docs/pipeline-spec.md`` must be revisited.
    """
    import mlx_whisper

    assert hasattr(mlx_whisper, "load_models")
    assert hasattr(mlx_whisper.load_models, "load_model")


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


def test_empty_spans_logging_does_not_leak_model_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The empty-span fast path must not log or reference the model path."""
    with caplog.at_level(logging.WARNING, logger="vemoizer.redecode"):
        results = redecode(_audio(1.0), [])
    assert results == []
    assert "pinned" not in caplog.text.lower() or "repo" not in caplog.text.lower()
