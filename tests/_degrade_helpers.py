"""Shared fixtures for the far-field degrade test modules (issue #154).

Kept as an explicit import (like ``_cli_helpers``) so the test modules use the
helpers by name and ``conftest.py`` stays configuration only.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE, write_wav


def make_corpus(root: Path, stems: tuple[str, ...] = ("a", "b")) -> Path:
    """A tiny stem-paired corpus (0.5 s sine per clip, reference ``.txt``)."""
    corpus = root / "corpus"
    corpus.mkdir()
    for stem in stems:
        x = (0.3 * np.sin(np.arange(SAMPLE_RATE // 2) / 10.0)).astype(np.float32)
        write_wav(corpus / f"{stem}.wav", x)
        (corpus / f"{stem}.txt").write_text(f"hello {stem}\n", encoding="utf-8")
    return corpus


def read_wav_float(path: Path) -> np.ndarray:
    """Read a 16-bit PCM WAV as float32 in [-1, 1) (a plain ingest stand-in)."""
    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32767.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))
