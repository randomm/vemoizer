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
    # Part markers for a multi-part (concatenated) group: a list of
    # {"offset": float, "label": str} dicts, one per source part. Present
    # ONLY for multi-part groups (issue #77); single-file results carry no
    # part_markers key at all. Injected by the batch layer, not by a
    # transcriber backend.
    part_markers: list[dict[str, Any]]
    language: str
    # Per-window language distribution (issue #147, display only): e.g.
    # "fi 29/30, en 1/30". Present when at least one window reported a
    # language; absent when no window reported one (empty decode).
    language_summary: str
    # Windows that contained speech but returned 0 segments even after the
    # prompt-free retry (issue #152). Each entry is a (start_s, end_s)
    # tuple — the window's time range on the recording timeline. Present
    # only when at least one window was lost; absent on a clean decode.
    # Consumed only by the quality report; backends may omit it.
    #
    # Unlike language_summary (present-on-clean), lost_windows is
    # absent-on-clean: the key is omitted when no window was lost, so
    # callers should use .get('lost_windows') or isinstance checks rather
    # than assuming its presence.
    lost_windows: list[tuple[float, float]]
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
