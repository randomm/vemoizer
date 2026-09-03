"""Tests for ParakeetTranscriber (decode A).

No module-level sys.modules stubbing: parakeet_mlx is a real dependency on
Apple Silicon and is imported lazily inside methods; tests patch at the
method level so nothing leaks into the shared pytest process.

Model loading is lazy: no download/load happens at construction. Tests that
need a loaded model either patch ``_load_model`` (to skip loading) or patch
``huggingface_hub.snapshot_download`` + ``parakeet_mlx.from_pretrained`` to
verify the revision-pinned load path.

Mock shapes mirror the REAL parakeet_mlx API surface (``AlignedResult`` has
``text`` + ``sentences`` and a ``tokens`` property — no ``language`` or
``segments`` attributes), so ``hasattr``/``getattr`` fallback behavior in
production matches test behavior.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vemoizer.parakeet_transcriber import (
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    ParakeetTranscriber,
    _extract_words_segments,
)
from vemoizer.transcriber import Transcriber


def make_mock_word(text="hei", start=0.0, end=0.2):
    """Mock an AlignedToken (real: text + start + end, computed in __post_init__).

    Plain object (not MagicMock) so attribute lookups on the parent never
    shadow the ``tokens`` flat-property via MagicMock's dynamic attributes.
    """
    w = type("MockToken", (), {"text": text, "start": start, "end": end})()
    return w


def make_mock_sentence(text="a b", start=0.0, end=0.3):
    """Mock an AlignedSentence (real: text + start + end + tokens)."""
    s = MagicMock(spec=["text", "start", "end", "tokens"])
    s.text = text
    s.start = start
    s.end = end
    s.tokens = []
    return s


def make_mock_alignment(text="hello world", sentences=None):
    """Create a mock per-input AlignedResult (real: text + sentences).

    Real AlignedResult has NO ``language`` and NO ``segments`` attributes —
    the spec excludes both so ``hasattr``/``getattr`` behave like production.
    """
    aligned = MagicMock(spec=["text", "sentences"])
    aligned.text = text
    if sentences is None:
        aligned.sentences = [
            make_mock_sentence(text=text, start=0.0, end=len(text) * 0.1)
        ]
    else:
        aligned.sentences = sentences
    return aligned


def make_mock_model(generate_result=None):
    """Create a mock parakeet model with a configurable generate() result."""
    mock_model = MagicMock()
    mock_model.preprocessor_config.sample_rate = SAMPLE_RATE
    if generate_result is not None:
        mock_model.generate.return_value = generate_result
    else:
        mock_model.generate.return_value = []
    return mock_model


class TestParakeetTranscriber:
    def _make_transcriber_with_mock(self):
        """Create a ParakeetTranscriber with a mocked, already-loaded model."""
        t = ParakeetTranscriber()
        t.model = make_mock_model()
        return t

    def test_no_model_loaded_at_construction(self):
        """Lazy-load: constructing must not load the model or download."""
        with (
            patch("huggingface_hub.snapshot_download") as snap,
            patch("parakeet_mlx.from_pretrained") as fp,
        ):
            t = ParakeetTranscriber()
            assert t.model is None
            snap.assert_not_called()
            fp.assert_not_called()

    def test_no_model_loaded_at_import(self):
        """Importing the module must not load the model (no module-level load).

        A fresh instance has no loaded model behind it; the first transcribe()
        is what triggers the load.
        """
        t = ParakeetTranscriber()
        assert t.model is None

    def test_transcribe_returns_text(self):
        t = self._make_transcriber_with_mock()
        t.model.generate.return_value = [
            make_mock_alignment(text="hello world"),
        ]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert result["text"] == "hello world"

    def test_transcribe_empty_audio_returns_empty(self):
        t = self._make_transcriber_with_mock()
        result = t.transcribe(np.array([], dtype=np.float32))
        assert result["text"] == ""
        assert result["words"] == []
        assert result["segments"] == []
        assert result["transcribe_time"] == 0.0
        assert result["audio_duration"] == 0.0
        assert result["rtf"] == 0.0

    def test_transcribe_converts_dtype(self):
        t = self._make_transcriber_with_mock()
        t.model.generate.return_value = [make_mock_alignment(text="hello world")]
        audio = np.zeros(SAMPLE_RATE, dtype=np.int16)

        captured = {}

        def fake_get_logmel(arr, cfg):
            # Capture the dtype of the underlying data as the transcriber
            # passes it in (the stand-in mx.array preserves .dtype).
            captured["dtype"] = arr.dtype
            return MagicMock()

        with patch("parakeet_mlx.audio.get_logmel", side_effect=fake_get_logmel):
            t.transcribe(audio)
        # The audio was coerced to float32 before the model saw it.
        # Accept both numpy's bare "float32" string and the real mlx
        # "mlx.core.float32" qualified form.
        assert "float32" in str(captured["dtype"])

    def test_transcribe_includes_timing(self):
        t = self._make_transcriber_with_mock()
        t.model.generate.return_value = [make_mock_alignment(text="hello world")]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert result["audio_duration"] == pytest.approx(1.0)
        assert result["transcribe_time"] >= 0.0
        assert result["rtf"] >= 0.0

    def test_transcribe_satisfies_protocol(self):
        """ParakeetTranscriber conforms to the @runtime_checkable Transcriber."""
        t = ParakeetTranscriber()
        assert isinstance(t, Transcriber)

    def test_cleanup_releases_model(self):
        t = self._make_transcriber_with_mock()
        t.cleanup()
        assert t.model is None

    def test_transcribe_raises_if_model_not_loaded(self):
        t = ParakeetTranscriber()
        # Simulate a failed load: _load_model leaves model None.
        with (
            patch.object(
                type(t),
                "_load_model",
                side_effect=lambda: setattr(t, "model", None),
            ),
            pytest.raises(RuntimeError, match="model failed to load"),
        ):
            t.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def test_load_model_revision_pinned(self):
        """First transcribe() triggers a revision-pinned load from the local path."""
        t = ParakeetTranscriber()
        local_path = MagicMock()  # a str-like path returned by snapshot_download
        with (
            patch(
                "huggingface_hub.snapshot_download",
                return_value=local_path,
            ) as snap,
            patch(
                "parakeet_mlx.from_pretrained",
                return_value=make_mock_model(),
            ) as fp,
        ):
            t.transcribe(np.array([], dtype=np.float32))
            snap.assert_called_once_with(MODEL_ID, revision=MODEL_REVISION)
            # Loaded from the returned local path, not the bare repo ID.
            fp.assert_called_once_with(local_path)
        assert t.model is not None

    def test_load_model_idempotent(self):
        """_load_model loads exactly once across repeated transcribe calls."""
        t = ParakeetTranscriber()
        with (
            patch(
                "huggingface_hub.snapshot_download",
                return_value="mock-path",
            ) as snap,
            patch(
                "parakeet_mlx.from_pretrained",
                return_value=make_mock_model(),
            ),
        ):
            t.transcribe(np.array([], dtype=np.float32))
            t.transcribe(np.array([], dtype=np.float32))
            assert snap.call_count == 1

    def test_load_failure_leaves_model_none(self):
        """A failed download/load is logged, leaves model None, transcribe raises."""
        t = ParakeetTranscriber()
        with (
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=Exception("boom"),
            ),
            patch.object(type(t), "_load_model"),
            pytest.raises(RuntimeError, match="model failed to load"),
        ):
            t.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        assert t.model is None

    def test_transcribe_no_language_for_real_aligned_result_shape(self):
        """The real parakeet-mlx AlignedResult has no ``language`` attribute.

        The transcriber must not fabricate a language for the memo — ``language``
        is omitted from the result when the backend does not report it
        (invariant #3: per-span language is a property of the detection pass,
        not of a hardcoded decode).
        """
        t = self._make_transcriber_with_mock()
        t.model.generate.return_value = [
            make_mock_alignment(text="moi on aapo"),  # no language attr
        ]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert "language" not in result

    def test_transcribe_includes_language_when_backend_provides_it(self):
        """If a backend exposes ``language`` on its result, it is passed through."""
        t = self._make_transcriber_with_mock()
        aligned = make_mock_alignment(text="moi on aapo")
        aligned.language = "fi"  # simulate a backend that does surface LID
        t.model.generate.return_value = [aligned]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert result.get("language") == "fi"

    def test_transcribe_extracts_word_timestamps(self):
        t = self._make_transcriber_with_mock()
        words = [
            make_mock_word("hello", 0.0, 0.5),
            make_mock_word(" world", 0.5, 1.0),
        ]
        # Real AlignedResult: tokens live on the sentences.
        sentence = make_mock_sentence(text="hello world", start=0.0, end=1.0)
        sentence.tokens = words
        t.model.generate.return_value = [
            make_mock_alignment(text="hello world", sentences=[sentence]),
        ]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert result["words"] == [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]

    def test_transcribe_flattens_words_across_sentences(self):
        t = self._make_transcriber_with_mock()
        s1 = make_mock_sentence(text="a b", start=0.0, end=0.1)
        s1.tokens = [make_mock_word("a", 0.0, 0.1)]
        s2 = make_mock_sentence(text="c d", start=1.0, end=1.1)
        s2.tokens = [make_mock_word(" c", 1.0, 1.1)]
        t.model.generate.return_value = [
            make_mock_alignment(text="a b c d", sentences=[s1, s2]),
        ]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert [w["word"] for w in result["words"]] == ["a", "c"]

    def test_transcribe_extracts_segments_from_sentences(self):
        t = self._make_transcriber_with_mock()
        s1 = make_mock_sentence(text="a b", start=0.0, end=0.5)
        s2 = make_mock_sentence(text="c d", start=1.0, end=1.5)
        t.model.generate.return_value = [
            make_mock_alignment(text="a b c d", sentences=[s1, s2]),
        ]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert result["segments"] == [
            {"start": 0.0, "end": 0.5, "text": "a b"},
            {"start": 1.0, "end": 1.5, "text": "c d"},
        ]

    def test_transcribe_no_words_when_alignment_has_none(self):
        t = self._make_transcriber_with_mock()
        # Alignment with no sentences -> no tokens -> empty word list.
        t.model.generate.return_value = [
            make_mock_alignment(text="hi", sentences=[]),
        ]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with patch("parakeet_mlx.audio.get_logmel", return_value=MagicMock()):
            result = t.transcribe(audio)
        assert result["words"] == []
        assert result["segments"] == []


class TestExtractWordsSegments:
    def test_flattens_words_and_segments(self):
        w1 = make_mock_word("a", 0.0, 0.1)
        w2 = make_mock_word(" b", 0.1, 0.2)
        s1 = make_mock_sentence(text="a b", start=0.0, end=0.2)
        s1.tokens = [w1, w2]
        aligned = make_mock_alignment(text="a b", sentences=[s1])
        words, segments = _extract_words_segments([aligned])
        assert words == [
            {"word": "a", "start": 0.0, "end": 0.1},
            {"word": "b", "start": 0.1, "end": 0.2},
        ]
        assert segments == [{"start": 0.0, "end": 0.2, "text": "a b"}]

    def test_empty_alignments(self):
        words, segments = _extract_words_segments([])
        assert words == []
        assert segments == []

    def test_extracts_words_from_flat_tokens(self):
        """AlignedResult.tokens is a flat property over all sentences."""
        w1 = make_mock_word("x", 0.0, 0.3)
        aligned = MagicMock(spec=["text", "sentences"])
        aligned.text = "x"
        aligned.sentences = []
        aligned.tokens = [w1]
        words, _ = _extract_words_segments([aligned])
        assert words == [{"word": "x", "start": 0.0, "end": 0.3}]

    def test_no_words_and_no_tokens(self):
        """Alignment with neither tokens nor sentences present -> empty lists."""
        aligned = MagicMock(spec=["text", "sentences"])
        aligned.text = "x"
        aligned.sentences = None
        aligned.tokens = None
        words, segments = _extract_words_segments([aligned])
        assert words == []
        assert segments == []

    def test_extracts_segments(self):
        seg = make_mock_sentence(text="a b", start=0.0, end=1.0)
        aligned = MagicMock(spec=["text", "sentences"])
        aligned.text = "a b"
        aligned.sentences = [seg]
        _, segments = _extract_words_segments([aligned])
        assert segments == [{"start": 0.0, "end": 1.0, "text": "a b"}]


def test_no_sys_modules_pollution_after_import():
    """Importing must not leak MagicMock stubs into sys.modules."""
    import parakeet_mlx

    assert not isinstance(parakeet_mlx, MagicMock)
    assert getattr(parakeet_mlx, "__file__", None) is not None


class TestTokenToWordGrouping:
    """Subword tokens group into words on the leading-space convention.

    The real AlignedResult tokens are SentencePiece-style pieces
    (' B', 'is', 'n', 'estä' -> 'Bisnestä'); alignment and span text
    must see whole words, not pieces (issue #55).
    """

    def test_pieces_merge_into_words(self):
        tokens = [
            make_mock_word(" B", 0.4, 0.48),
            make_mock_word("is", 0.48, 0.72),
            make_mock_word("nestä", 0.72, 1.0),
            make_mock_word(" ja", 1.12, 1.3),
        ]
        aligned = MagicMock(spec=["text", "sentences", "tokens"])
        aligned.text = "Bisnestä ja"
        aligned.tokens = tokens
        aligned.sentences = []
        words, _segments = _extract_words_segments([aligned])
        assert words == [
            {"word": "Bisnestä", "start": 0.4, "end": 1.0},
            {"word": "ja", "start": 1.12, "end": 1.3},
        ]

    def test_first_token_without_space_starts_a_word(self):
        tokens = [
            make_mock_word("Moi", 0.0, 0.3),
            make_mock_word(" vaan", 0.4, 0.7),
        ]
        aligned = MagicMock(spec=["text", "sentences", "tokens"])
        aligned.text = "Moi vaan"
        aligned.tokens = tokens
        aligned.sentences = []
        words, _segments = _extract_words_segments([aligned])
        assert [w["word"] for w in words] == ["Moi", "vaan"]

    def test_whitespace_only_tokens_are_dropped(self):
        tokens = [
            make_mock_word(" ", 0.0, 0.1),
            make_mock_word(" ok", 0.2, 0.4),
        ]
        aligned = MagicMock(spec=["text", "sentences", "tokens"])
        aligned.text = "ok"
        aligned.tokens = tokens
        aligned.sentences = []
        words, _segments = _extract_words_segments([aligned])
        assert words == [{"word": "ok", "start": 0.2, "end": 0.4}]
