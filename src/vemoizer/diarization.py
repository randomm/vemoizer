"""Optional speaker diarization (issue #13, opt-in ``--diarize``).

Uses ``pyannote.audio==4.0.7`` with the
``pyannote/speaker-diarization-community-1`` weights. The weights are
**CC-BY-4.0-licensed and gated on HuggingFace** — the user must accept the
license form and provide an access token before first use. Attribution is
mandatory under CC-BY-4.0 and is exposed via :data:`ATTRIBUTION`.

pyannote is lazily imported inside :func:`_load_pipeline` so the default
(no-``--diarize``) path never touches it and never needs it installed.
Device selection: MPS is attempted first (fixed in pyannote PR 1546); on
any load/inference exception we fall back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: HuggingFace repo for the diarization weights (CC-BY-4.0, gated).
DIARIZATION_REPO_ID = "pyannote/speaker-diarization-community-1"

#: Mandatory CC-BY-4.0 attribution (weights license, not code license).
ATTRIBUTION = (
    "Speaker diarization: pyannote/speaker-diarization-community-1 "
    "(weights licensed under CC-BY-4.0, gated on HuggingFace; "
    "user accepted the license form and supplied an access token)."
)

#: Sample rate of the internal audio contract (AGENTS.md invariant #6).
_CONTRACT_SAMPLE_RATE = 16000

#: Environment variable holding the HuggingFace access token for the gated repo.
_HF_TOKEN_ENV = "HF_TOKEN"


@dataclass(frozen=True)
class DiarizationResult:
    """Speaker-labelled time segments covering the recording."""

    segments: list[tuple[float, float, str]]  # (start_s, end_s, speaker_label)


def _load_pipeline(device: str) -> object:
    """Lazily import pyannote and build the community pipeline on *device*."""
    import os

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_REPO_ID,
        use_auth_token=os.environ.get(_HF_TOKEN_ENV),
    )
    pipeline.to(torch.device(device))
    return pipeline


def diarize(audio: np.ndarray, *, device: str = "auto") -> DiarizationResult:
    """Run speaker diarization over 16 kHz mono float32 *audio*.

    ``device="auto"`` tries MPS first (Apple Silicon) and falls back to CPU
    on any load/inference exception. ``device`` may also be an explicit
    torch device name (e.g. ``"cpu"`` or ``"mps"``).
    """
    waveforms = {
        "audio": audio.astype(np.float32),
        "sample_rate": _CONTRACT_SAMPLE_RATE,
    }

    if device == "auto":
        try:
            pipeline = _load_pipeline("mps")
            diarization = pipeline(waveforms)  # ty: ignore[call-non-callable]
        except Exception:
            pipeline = _load_pipeline("cpu")
            diarization = pipeline(waveforms)  # ty: ignore[call-non-callable]
    else:
        pipeline = _load_pipeline(device)
        diarization = pipeline(waveforms)  # ty: ignore[call-non-callable]

    segments: list[tuple[float, float, str]] = [
        (turn.start, turn.end, turn.speaker)
        for turn in diarization.itertracks(yield_label=True)
    ]
    return DiarizationResult(segments=segments)
