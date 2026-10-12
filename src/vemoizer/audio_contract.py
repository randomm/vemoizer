"""The 16 kHz mono float32 internal audio contract (AGENTS.md invariant #6).

Single home for the contract constants so every stage (ingest, VAD, decode,
alignment, diarization) reads the same tokens. Do not restate these
constants in other modules — import from here.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

#: Canonical internal sample rate: 16 kHz.
SAMPLE_RATE = 16_000

#: Sample rates silero-vad natively supports (8/16 kHz).
SUPPORTED_SAMPLE_RATES: tuple[int, ...] = (8000, 16000)


def write_wav(path: Path, x: np.ndarray) -> Path:
    """Write *x* as 16 kHz mono 16-bit PCM WAV (deterministic bytes).

    16-bit is an eval-only quantisation of the float32 contract; see
    docs/pipeline-spec.md.
    """
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return path
