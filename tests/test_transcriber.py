"""Tests for the Transcriber protocol (backend seam)."""

from typing import Protocol

import numpy as np

from vemoizer.transcriber import Transcriber, TranscriptionResult


class _MockTranscriber:
    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        return {"text": "hello"}

    def cleanup(self) -> None:
        pass


class _NotATranscriber:
    pass


def test_transcriber_protocol_is_protocol() -> None:
    """Transcriber is a typing.Protocol."""
    assert hasattr(Transcriber, "__protocol__") or Protocol in Transcriber.__bases__


def test_transcriber_protocol_methods() -> None:
    """Protocol defines transcribe and cleanup."""
    assert hasattr(Transcriber, "transcribe")
    assert hasattr(Transcriber, "cleanup")


def test_protocol_runtime_checkable() -> None:
    """Protocol is @runtime_checkable — isinstance() works."""
    mock = _MockTranscriber()
    assert isinstance(mock, Transcriber)


def test_non_conforming_class_fails_isinstance() -> None:
    """Object missing protocol methods fails isinstance check."""
    obj = _NotATranscriber()
    assert not isinstance(obj, Transcriber)


def test_parakeet_transcriber_satisfies_protocol() -> None:
    """ParakeetTranscriber (decode A) conforms to the protocol."""
    from vemoizer.parakeet_transcriber import ParakeetTranscriber

    assert isinstance(ParakeetTranscriber(), Transcriber)


def test_transcription_result_type() -> None:
    """TranscriptionResult is a TypedDict with text required, rest optional."""
    import typing

    assert issubclass(TranscriptionResult, dict)
    fields = typing.get_type_hints(TranscriptionResult)
    assert "text" in fields
    # total=False makes non-text keys optional.
    assert TranscriptionResult.__total__ is False
