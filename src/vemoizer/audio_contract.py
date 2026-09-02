"""The 16 kHz mono float32 internal audio contract (AGENTS.md invariant #6).

Single home for the contract constants so every stage (ingest, VAD, decode,
alignment, diarization) reads the same tokens. Do not restate these
constants in other modules — import from here.
"""

from __future__ import annotations

#: Canonical internal sample rate: 16 kHz.
SAMPLE_RATE = 16_000

#: Sample rates silero-vad natively supports (8/16 kHz).
SUPPORTED_SAMPLE_RATES: tuple[int, ...] = (8000, 16000)
