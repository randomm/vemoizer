"""Shared fixtures for the whisper meeting-decode tests (issue #152, FIX 1).

Split out of ``test_whisper_transcriber.py`` so the lost-window retry tests
live in ``test_whisper_lost_windows.py`` without duplicating the helpers
(AGENTS.md: do not duplicate helper code; keep both files under the 800-line
test cap).
"""

from __future__ import annotations

import numpy as np


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


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def _speech_audio(seconds: float) -> np.ndarray:
    """Non-silent audio (sine wave) that passes the energy check."""
    t = np.linspace(0, seconds, int(seconds * 16_000), endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
