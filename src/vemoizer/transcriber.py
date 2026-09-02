"""Transcriber Protocol for speech-to-text backends.

The ``Transcriber`` Protocol is the backend seam of the consensus pipeline
(project invariant #2): every ASR model (Parakeet decode A, Canary decode B,
Whisper re-decode) is reached through this interface, which is what makes the
dual-decode + consensus architecture and the eval harness possible. Pipeline
and CLI code must never call a model library directly.
"""

from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np


class _TranscriptionBase(TypedDict):
    """Base TypedDict for transcription results with required fields."""

    text: str  # always required


class TranscriptionResult(_TranscriptionBase, total=False):
    """TypedDict for transcription results.

    The 'text' field is always required. Other fields are optional.

    Field contract (consumed by the alignment stage, ticket 6):

    - ``words``: list of ``{"word": str, "start": float, "end": float}``
      dicts in time order — one entry per recognized word/token.
    - ``segments``: list of ``{"start": float, "end": float, "text": str}``
      dicts in time order — sentence-level chunks of the recording.
    - ``language``: ISO 639-1 code of the utterance, present only when the
      backend actually reports one (per-utterance, never a hardcoded file
      language — invariant #3).
    """

    words: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    language: str
    transcribe_time: float
    audio_duration: float
    rtf: float


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for speech-to-text transcriber implementations.

    This Protocol is runtime checkable, so isinstance() can be used to verify
    that an implementation conforms to the interface.
    """

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz typically)
            **kwargs: Additional backend-specific parameters

        Returns:
            TranscriptionResult dict with at least 'text' key
        """
        ...

    def cleanup(self) -> None:
        """Release resources and cleanup."""
        ...
